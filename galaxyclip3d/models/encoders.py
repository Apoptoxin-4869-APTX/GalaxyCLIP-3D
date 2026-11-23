"""Point-cloud encoders: a cloud of N points with F values each, (B, N, F), becomes one feature vector, (B, D).

All encoders ignore the order of the points, and use LayerNorm rather than BatchNorm: a masked modality arrives as an
all-zero cloud, which must not leak into statistics shared across the batch.

- PointNetEncoder: the same small network on every point, then max and mean pooling (Qi et al. 2017). A fast
  baseline that sees each point alone, so it cannot use local structure. `checkpoint` recomputes each layer in the
  backward pass instead of storing its intermediate results: the same output and gradients in much less GPU memory
  (Stage E at batch 256: 27.7 instead of 58.7 GiB), for a training step about a quarter longer.
- PatchTransformerEncoder: the Point-MAE / Point-BERT design (Pang et al. 2022, Yu et al. 2022). Farthest-point
  sampling picks `patches` centres, each centre takes its `patch_size` nearest points, a small PointNet turns each
  patch (positions relative to its centre, plus the other values) into a token, a transformer mixes the tokens with
  a learned embedding of the centre positions, and a class token plus the mean token give the output. With
  `centres="radial"` the centres are picked in radially equalised coordinates (radial_equalise), so they follow the
  points' density: plain farthest-point sampling puts only about 2% of 128 centres in the inner half of a star cloud.
- PointNet2Encoder: PointNet++ (Qi et al. 2017), for local structure without a transformer. Set-abstraction levels
  each pick centres (radially equalised farthest-point sampling by default), group every centre with its k nearest
  points and summarise each group with a small PointNet; a PointNet over the last level's centres gives the output.
  The k nearest points (not a fixed radius) adapt the groups to the dense centre and the sparse outskirts; positions in
  a group are divided by its radius (distance to the k-th point), whose logarithm is kept, so density is not lost; and
  every point also carries its values relative to the centre's (local motions, gradients).
- PointNeXtEncoder: PointNeXt (Qian et al. 2022), a deeper PointNet++ with residual blocks. A stem layer on every
  point, then stages of one set-abstraction block (centres, k nearest points, one layer per point, max over the group,
  plus the centre's own features as a shortcut) and InvResMLP blocks (one layer on each point's k nearest neighbours,
  max, then two pointwise layers widened 4x and back, added to the input). Adapted as PointNet2Encoder: radially
  equalised farthest-point centres, k nearest points with positions divided by the group radius and its logarithm as a
  feature, LayerNorm after the max (masked clouds are all zeros), GELU; max and mean pooling into the same head as the
  other encoders. Defaults: PointNeXt-B's block pattern (1, 2, 1, 1), 4,096 points reduced to 1,024, 512, 256 and 128
  centres, widths capped at 256 so an encoder has about as many parameters as the PointNet and PointNet++.
- VNPointNetEncoder: a PointNet built from Vector Neurons (Deng et al. 2021), exactly unchanged by any rotation or
  mirroring of the cloud, with no frame to estimate. Each point's position and velocity enter as two 3D vectors
  (`vectors=1` without velocities) and its other values, plus its distance, speed, radial and tangential velocity, as
  scalars. Every layer (VNBlock) mixes the vector channels linearly with the same weights for x, y and z and no bias,
  feeds their lengths to the scalar channels (as in GVP, Jing et al. 2021), applies the Vector Neurons LeakyReLU along
  learned directions, divides a point's vectors by their root-mean-square length and scales each by a sigmoid of the
  new scalars. `checkpoint` recomputes each layer in the backward pass instead of storing its intermediate results:
  the many vector-sized intermediates otherwise take about three times the PointNet's memory. Pooling: each vector
  channel's softmax-weighted mean over the points, the weights from the dot product with a learned direction (a soft
  version of Vector Neurons' max pooling, whose hard pick gives the direction no gradient and jumps between points
  that tie to within rounding), and its plain mean, read off as dot products with three vectors computed from them
  (Vector Neurons' invariant layer) plus their lengths, and the scalars' max and mean, into the same head as the other
  encoders. Only lengths and dot products of vectors are used, never a cross product, so mirror images give the same
  output too. A vector channel holds three numbers per point, so the default widths are about a third of the
  PointNet's.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn


def invariant_features(cloud: torch.Tensor, velocities: bool = True) -> torch.Tensor:
    """(..., N, 4) per-point numbers that stay the same when a cloud is rotated or mirrored, from its positions
    (columns 0-2) and velocities (3-5): distance from the centre, speed, radial velocity and tangential speed.
    (..., N, 1), the distance only, with velocities=False. All-zero (masked) clouds give zeros."""
    pos = cloud[..., 0:3].float()
    r = pos.norm(dim=-1, keepdim=True)
    if not velocities:
        return r
    vel = cloud[..., 3:6].float()
    rr = r.clamp_min(1e-6)
    radial = (pos * vel).sum(-1, keepdim=True) / rr
    tangential = torch.linalg.cross(pos, vel, dim=-1).norm(dim=-1, keepdim=True) / rr
    return torch.cat([r, vel.norm(dim=-1, keepdim=True), radial, tangential], dim=-1)


def mlp(dims: list[int], final_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or final_act:
            layers += [nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*layers)


def layer_groups(seq: nn.Sequential) -> list[list[nn.Module]]:
    """The modules of an mlp() split before each Linear: one group per layer (Linear, LayerNorm, GELU)."""
    groups: list[list[nn.Module]] = []
    for module in seq:
        if isinstance(module, nn.Linear) or not groups:
            groups.append([])
        groups[-1].append(module)
    return groups


def run_layers(modules: list[nn.Module], h: torch.Tensor) -> torch.Tensor:
    for module in modules:
        h = module(h)
    return h


class PointNetEncoder(nn.Module):
    def __init__(self, in_dim: int, widths=(64, 128, 256, 512), out_dim: int = 512, dropout: float = 0.0,
                 checkpoint: bool = False):
        super().__init__()
        self.checkpoint = checkpoint             # recompute each layer in the backward pass instead of storing it
        self.point_mlp = mlp([in_dim, *widths])
        last = widths[-1] if len(widths) else in_dim
        self.head = nn.Sequential(nn.Linear(2 * last, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and torch.is_grad_enabled():
            h = x
            for modules in layer_groups(self.point_mlp):
                h = torch.utils.checkpoint.checkpoint(run_layers, modules, h, use_reentrant=False)
        else:
            h = self.point_mlp(x)
        return self.head(torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1))


def farthest_point_sample(xyz: torch.Tensor, n: int) -> torch.Tensor:
    """(B, n) indices of points spread over each cloud. It starts from the point farthest from the cloud's centroid,
    so the result does not depend on the order of the points (barring exact ties)."""
    b, npts, _ = xyz.shape
    batch = torch.arange(b, device=xyz.device)
    far = ((xyz - xyz.mean(dim=1, keepdim=True)) ** 2).sum(-1).argmax(dim=1)
    dist = torch.full((b, npts), float("inf"), device=xyz.device, dtype=xyz.dtype)
    idx = torch.empty(b, n, dtype=torch.long, device=xyz.device)
    for i in range(n):
        idx[:, i] = far
        dist = torch.minimum(dist, ((xyz - xyz[batch, far][:, None, :]) ** 2).sum(-1))
        far = dist.argmax(dim=1)
    return idx


def radial_equalise(xyz: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) positions with each point's distance from the centre replaced by cdf(r)^(1/3), its direction kept:
    the cloud becomes uniform in volume, so points spread evenly in these coordinates follow the density."""
    r = xyz.norm(dim=-1)
    cdf = (r.argsort(dim=1).argsort(dim=1).to(xyz.dtype) + 0.5) / r.shape[1]
    return xyz / r.clamp_min(1e-12)[..., None] * cdf[..., None] ** (1 / 3)


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, F), idx (B, ...) -> (B, ..., F)."""
    b = x.shape[0]
    flat = idx.reshape(b, -1)
    out = torch.gather(x, 1, flat[..., None].expand(-1, -1, x.shape[-1]))
    return out.reshape(*idx.shape, x.shape[-1])


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, width: int = 384, depth: int = 6, heads: int = 6, patches: int = 128,
                 patch_size: int = 32, dropout: float = 0.0, out_dim: int = 512, centres: str = "fps"):
        super().__init__()
        if centres not in ("fps", "radial"):
            raise ValueError(f"centres must be fps or radial, got {centres!r}")
        self.patches, self.patch_size, self.centres = patches, patch_size, centres
        self.local1 = mlp([in_dim, 128, 256])
        self.local2 = mlp([512, 512, width], final_act=False)
        self.pos = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(2 * width, out_dim)
        self.out_dim = out_dim

    def tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(patch tokens (B, G, width), centre positions (B, G, 3))."""
        xyz = x[..., :3]
        with torch.no_grad():
            pts = xyz.float()
            centre_idx = farthest_point_sample(radial_equalise(pts) if self.centres == "radial" else pts, self.patches)
            centres = gather_points(xyz, centre_idx)                                   # (B, G, 3)
            k = min(self.patch_size, x.shape[1])
            near = torch.cdist(centres.float(), xyz.float()).topk(k, dim=-1, largest=False).indices   # (B, G, k)
        group = gather_points(x, near)                                                 # (B, G, k, F)
        group = torch.cat([group[..., :3] - centres[:, :, None, :], group[..., 3:]], dim=-1)
        f = self.local1(group)                                                         # (B, G, k, 256)
        f = torch.cat([f, f.max(dim=2, keepdim=True).values.expand_as(f)], dim=-1)    # (B, G, k, 512)
        return self.local2(f).max(dim=2).values, centres

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok, centres = self.tokens(x)
        b = x.shape[0]
        h = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)
        h = h + torch.cat([self.cls_pos.expand(b, -1, -1), self.pos(centres)], dim=1)
        h = self.norm(self.blocks(h))
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1))


class PointNet2Encoder(nn.Module):
    SAMPLING = ("radial", "fps")

    def __init__(self, in_dim: int, out_dim: int = 512, centres=(512, 128), neighbours=(32, 32),
                 widths=((64, 64, 128), (128, 128, 256)), global_widths=(128, 256, 512, 1024),
                 sampling: str = "radial", dropout: float = 0.0):
        super().__init__()
        if sampling not in self.SAMPLING:
            raise ValueError(f"sampling must be one of {self.SAMPLING}, got {sampling!r}")
        if not centres or not len(centres) == len(neighbours) == len(widths):
            raise ValueError("centres, neighbours and widths need one entry per level")
        if min(centres) < 1 or min(neighbours) < 1 or any(b >= a for a, b in zip(centres[:-1], centres[1:])):
            raise ValueError(f"centres must be positive and fall from level to level, neighbours positive: "
                             f"{list(centres)}, {list(neighbours)}")
        self.centres, self.neighbours, self.sampling = list(centres), list(neighbours), sampling
        self.levels = nn.ModuleList()
        c = in_dim
        for w in widths:
            self.levels.append(mlp([3 + 2 * c, *w]))     # relative position, values relative to the centre, values
            c = 2 * w[-1] + 4                            # max and mean over the group, centre position, log radius
        self.global_mlp = mlp([c, *global_widths])
        last = global_widths[-1] if len(global_widths) else c
        self.head = nn.Sequential(nn.Linear(2 * last, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz, f = x[..., :3], x                           # level 0: every input column, positions included
        for mlp_l, m, k in zip(self.levels, self.centres, self.neighbours):
            with torch.no_grad():
                pts = xyz.float()
                idx = farthest_point_sample(radial_equalise(pts) if self.sampling == "radial" else pts,
                                            min(m, pts.shape[1]))                            # (B, M)
                cxyz = gather_points(pts, idx)                                               # (B, M, 3)
                dist, near = torch.cdist(cxyz, pts).topk(min(k, pts.shape[1]), dim=-1, largest=False)  # (B, M, k)
                radius = dist[..., -1:].clamp_min(1e-4)                                      # (B, M, 1)
                rel = (gather_points(pts, near) - cxyz[:, :, None, :]) / radius[..., None]  # (B, M, k, 3)
            nbr = gather_points(f, near)                                                     # (B, M, k, C)
            ctr = gather_points(f, idx)[:, :, None, :]                                      # (B, M, 1, C)
            h = mlp_l(torch.cat([rel.to(nbr.dtype), nbr - ctr, nbr], dim=-1))               # (B, M, k, w)
            f = torch.cat([h.max(dim=2).values, h.mean(dim=2), cxyz.to(h.dtype), radius.log().to(h.dtype)], dim=-1)
            xyz = cxyz
        g = self.global_mlp(f)
        return self.head(torch.cat([g.max(dim=1).values, g.mean(dim=1)], dim=-1))


def knn_group(pts: torch.Tensor, centres: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each centre's (B, M, 3) k nearest points of pts (B, N, 3): indices (B, M, k), the group radius (distance to the
    k-th point, at least 1e-4; B, M, 1) and the points' positions relative to the centre divided by it (B, M, k, 3)."""
    with torch.no_grad():
        dist, near = torch.cdist(centres, pts).topk(min(k, pts.shape[1]), dim=-1, largest=False)
        radius = dist[..., -1:].clamp_min(1e-4)
        rel = (gather_points(pts, near) - centres[:, :, None, :]) / radius[..., None]
    return near, radius, rel


def group_inputs(f: torch.Tensor, near: torch.Tensor, radius: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """Per neighbour: relative position / radius, log radius, the neighbour's features ((B, M, k, 4 + C))."""
    nbr = gather_points(f, near)
    logr = radius.log()[:, :, None, :].expand(-1, -1, near.shape[-1], -1)
    return torch.cat([rel.to(nbr.dtype), logr.to(nbr.dtype), nbr], dim=-1)


class SetAbstraction(nn.Module):
    """PointNeXt's downsampling block: centres, their k nearest points, one (or two) layers per point, max over the
    group, plus the centre's own features through a linear shortcut; LayerNorm and GELU after the max."""

    def __init__(self, in_width: int, out_width: int, centres: int, neighbours: int, layers: int = 1,
                 sampling: str = "radial"):
        super().__init__()
        self.centres, self.k, self.sampling = centres, neighbours, sampling
        dims = [in_width + 4] + [out_width] * layers
        self.mlp = mlp(dims, final_act=False) if layers > 1 else nn.Linear(dims[0], out_width)
        self.shortcut = nn.Linear(in_width, out_width)
        self.norm = nn.LayerNorm(out_width)

    def forward(self, xyz: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            idx = farthest_point_sample(radial_equalise(xyz) if self.sampling == "radial" else xyz,
                                        min(self.centres, xyz.shape[1]))
            cxyz = gather_points(xyz, idx)
        near, radius, rel = knn_group(xyz, cxyz, self.k)
        h = self.mlp(group_inputs(f, near, radius, rel)).max(dim=2).values + self.shortcut(gather_points(f, idx))
        return cxyz, nn.functional.gelu(self.norm(h))


class InvResMLP(nn.Module):
    """PointNeXt's inverted residual block on a fixed set of points: one layer on each point's k nearest neighbours,
    max over them, LayerNorm and GELU, two pointwise layers (widened `expansion` times and back), added to the input."""

    def __init__(self, width: int, neighbours: int, expansion: int = 4):
        super().__init__()
        self.k = neighbours
        self.local = nn.Linear(width + 4, width)
        self.norm = nn.LayerNorm(width)
        self.pointwise = nn.Sequential(nn.Linear(width, expansion * width), nn.LayerNorm(expansion * width), nn.GELU(),
                                       nn.Linear(expansion * width, width), nn.LayerNorm(width))

    def forward(self, xyz: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        near, radius, rel = knn_group(xyz, xyz, self.k)
        h = nn.functional.gelu(self.norm(self.local(group_inputs(f, near, radius, rel)).max(dim=2).values))
        return nn.functional.gelu(f + self.pointwise(h))


class PointNeXtEncoder(nn.Module):
    SAMPLING = ("radial", "fps")

    def __init__(self, in_dim: int, out_dim: int = 512, width: int = 32, centres=(1024, 512, 256, 128),
                 neighbours=(32, 32, 32, 32), widths=(64, 128, 256, 256), blocks=(1, 2, 1, 1), expansion: int = 4,
                 sampling: str = "radial", dropout: float = 0.0):
        super().__init__()
        if sampling not in self.SAMPLING:
            raise ValueError(f"sampling must be one of {self.SAMPLING}, got {sampling!r}")
        if not centres or not len(centres) == len(neighbours) == len(widths) == len(blocks):
            raise ValueError("centres, neighbours, widths and blocks need one entry per stage")
        if (min(centres) < 1 or min(neighbours) < 1 or min(widths) < 1 or min(blocks) < 0 or width < 1
                or expansion < 1 or any(b >= a for a, b in zip(centres[:-1], centres[1:]))):
            raise ValueError(f"centres must be positive and fall from stage to stage, neighbours, widths, width and "
                             f"expansion positive, blocks not negative: {list(centres)}, {list(neighbours)}, "
                             f"{list(widths)}, {list(blocks)}")
        self.stem = nn.Sequential(nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU())
        self.stages = nn.ModuleList()
        prev = width
        for m, k, w, nb in zip(centres, neighbours, widths, blocks):
            self.stages.append(nn.ModuleList([SetAbstraction(prev, w, m, k, 1 if nb else 2, sampling),
                                              *[InvResMLP(w, k, expansion) for _ in range(nb)]]))
            prev = w
        self.head = nn.Sequential(nn.Linear(2 * prev, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz, f = x[..., :3].float(), self.stem(x)
        for stage in self.stages:
            xyz, f = stage[0](xyz, f)
            for block in stage[1:]:
                f = block(xyz, f)
        return self.head(torch.cat([f.max(dim=1).values, f.mean(dim=1)], dim=-1))


VN_EPS = 1e-6


def vn_length(v: torch.Tensor) -> torch.Tensor:
    """Lengths (..., C) of vector channels (..., 3, C), smooth at zero (masked clouds are all zeros)."""
    return torch.sqrt((v * v).sum(dim=-2) + VN_EPS)


def vn_leaky_relu(q: torch.Tensor, d: torch.Tensor, slope: float = 0.2) -> torch.Tensor:
    """Vector Neurons' LeakyReLU on vector channels (..., 3, C): where a vector q points away from its learned
    direction d, its part along d is removed except a fraction `slope`; elsewhere q is kept. Dot products only.
    Directions much shorter than the point's typical one (below a tenth of its root-mean-square length) fade out
    smoothly instead of acting with a direction made of rounding noise (Vector Neurons' fixed 1e-6 floor lets the
    noise of near-zero directions, common when parallel vectors cancel in a mix, through almost 1,000 times)."""
    dot = (q * d).sum(dim=-2, keepdim=True)
    dd = (d * d).sum(dim=-2, keepdim=True)
    return q - (1 - slope) * dot.clamp(max=0) / (dd + 1e-2 * dd.mean(dim=-1, keepdim=True) + VN_EPS) * d


def vn_rms(v: torch.Tensor) -> torch.Tensor:
    """(..., 1) the root-mean-square length of each point's vector channels (..., 3, C), smooth at zero."""
    return torch.sqrt((v * v).sum(dim=-2).mean(dim=-1, keepdim=True) + VN_EPS)


class VNBlock(nn.Module):
    """One layer on every point of vector channels (..., 3, Cv) and scalar channels (..., Cs): the vectors are mixed
    linearly (no bias, the same weights for x, y and z), their lengths join the scalars' input, and the mixed vectors
    go through the Vector Neurons LeakyReLU along directions mixed from the input vectors, are divided by their
    root-mean-square length (one factor per point, so short vectors stay short: dividing each channel by its own length
    would blow up the rounding noise of near-zero vectors, common when parallel vectors cancel in a mix) and scaled by
    a sigmoid gate from the new scalars. The LeakyReLU commutes with positive factors, so the division and the gate
    are applied together after it, which saves one vector-sized tensor per layer."""

    def __init__(self, vectors_in: int, scalars_in: int, vectors_out: int, scalars_out: int, slope: float = 0.2):
        super().__init__()
        self.mix = nn.Linear(vectors_in, vectors_out, bias=False)
        self.direction = nn.Linear(vectors_in, vectors_out, bias=False)
        self.scalar = nn.Sequential(nn.Linear(scalars_in + vectors_out, scalars_out), nn.LayerNorm(scalars_out),
                                    nn.GELU())
        self.gate = nn.Linear(scalars_out, vectors_out)
        self.slope = slope

    def forward(self, v: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mix(v)
        s = self.scalar(torch.cat([s, vn_length(h).to(s.dtype)], dim=-1))
        scale = torch.sigmoid(self.gate(s)) / vn_rms(h)
        return vn_leaky_relu(h, self.direction(v).to(h.dtype), self.slope) * scale.unsqueeze(-2).to(h.dtype), s


class VNPointNetEncoder(nn.Module):
    def __init__(self, in_dim: int, vectors: int = 2, widths=(32, 64, 128, 256), scalar_widths=(64, 128, 256, 512),
                 out_dim: int = 512, dropout: float = 0.0, slope: float = 0.2, checkpoint: bool = False):
        super().__init__()
        self.checkpoint = checkpoint             # recompute each layer in the backward pass instead of storing it
        if vectors not in (1, 2):
            raise ValueError(f"vectors must be 1 (positions) or 2 (positions and velocities), got {vectors}")
        if not widths or len(widths) != len(scalar_widths) or min(widths) < 1 or min(scalar_widths) < 1:
            raise ValueError(f"widths and scalar_widths need the same number of positive entries, got "
                             f"{list(widths)}, {list(scalar_widths)}")
        if in_dim < 3 * vectors:
            raise ValueError(f"{in_dim} input columns cannot hold {vectors} vector(s)")
        self.vectors = vectors
        cv, cs = vectors, in_dim - 3 * vectors + (4 if vectors == 2 else 1)    # + the points' invariant numbers
        self.blocks = nn.ModuleList()
        for wv, ws in zip(widths, scalar_widths):
            self.blocks.append(VNBlock(cv, cs, wv, ws, slope))
            cv, cs = wv, ws
        self.pool_direction = nn.Linear(cv, cv, bias=False)
        self.frame = nn.Linear(2 * cv, 3, bias=False)
        self.head = nn.Sequential(nn.Linear(2 * cv * 3 + 2 * cv + 2 * cs, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        k = 3 * self.vectors
        v = x[..., :k].reshape(b, n, self.vectors, 3).transpose(-1, -2)            # (B, N, 3, vectors)
        s = torch.cat([x[..., k:], invariant_features(x, self.vectors == 2).to(x.dtype)], dim=-1)
        for block in self.blocks:
            if self.checkpoint and torch.is_grad_enabled():
                v, s = torch.utils.checkpoint.checkpoint(block, v, s, use_reentrant=False)
            else:
                v, s = block(v, s)
        weight = torch.softmax((v * self.pool_direction(v)).sum(dim=-2).float(), dim=1)      # (B, N, C)
        soft_max = torch.einsum("bnc,bnic->bic", weight.to(v.dtype), v)                    # (B, 3, C)
        z = torch.cat([soft_max, v.mean(dim=1)], dim=-1)                                   # (B, 3, 2C)
        proj = torch.einsum("bic,bij->bcj", z, self.frame(z)).flatten(1)            # dot products with 3 vectors
        g = torch.cat([proj, vn_length(z).to(proj.dtype), s.max(dim=1).values.to(proj.dtype),
                       s.mean(dim=1).to(proj.dtype)], dim=-1)
        return self.head(g)


def make_encoder(kind: str, in_dim: int, out_dim: int, **kw) -> nn.Module:
    if kind == "pointnet":
        return PointNetEncoder(in_dim, kw.get("pointnet_widths", (64, 128, 256, 512)), out_dim, kw.get("dropout", 0.0),
                               checkpoint=kw.get("pointnet_checkpoint", False))
    if kind == "patch":
        return PatchTransformerEncoder(in_dim, kw.get("width", 384), kw.get("depth", 6), kw.get("heads", 6),
                                       kw.get("patches", 128), kw.get("patch_size", 32), kw.get("dropout", 0.0),
                                       out_dim, kw.get("patch_centres", "fps"))
    if kind == "pointnet2":
        return PointNet2Encoder(in_dim, out_dim, kw.get("sa_centres", (512, 128)), kw.get("sa_neighbours", (32, 32)),
                                kw.get("sa_widths", ((64, 64, 128), (128, 128, 256))),
                                kw.get("pointnet_widths", (128, 256, 512, 1024)), kw.get("sa_sampling", "radial"),
                                kw.get("dropout", 0.0))
    if kind == "pointnext":
        return PointNeXtEncoder(in_dim, out_dim, kw.get("nx_width", 32), kw.get("nx_centres", (1024, 512, 256, 128)),
                                kw.get("nx_neighbours", (32, 32, 32, 32)), kw.get("nx_widths", (64, 128, 256, 256)),
                                kw.get("nx_blocks", (1, 2, 1, 1)), kw.get("nx_expansion", 4),
                                kw.get("sa_sampling", "radial"), kw.get("dropout", 0.0))
    if kind == "vn_pointnet":
        return VNPointNetEncoder(in_dim, kw.get("vectors", 2), kw.get("vn_widths", (32, 64, 128, 256)),
                                 kw.get("vn_scalar_widths", (64, 128, 256, 512)), out_dim, kw.get("dropout", 0.0),
                                 checkpoint=kw.get("vn_checkpoint", False))
    raise ValueError(f"unknown encoder {kind!r} (pointnet, patch, pointnet2, pointnext or vn_pointnet)")
