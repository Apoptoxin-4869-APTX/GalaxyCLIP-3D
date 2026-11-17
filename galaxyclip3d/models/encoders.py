"""Point-cloud encoders: a cloud of N points with F values each, (B, N, F), becomes one feature vector, (B, D).

Both encoders ignore the order of the points, and use LayerNorm rather than BatchNorm: a masked modality arrives as an
all-zero cloud, which must not leak into statistics shared across the batch.

- PointNetEncoder: the same small network on every point, then max and mean pooling (Qi et al. 2017). A fast
  baseline that sees each point alone, so it cannot use local structure.
- PatchTransformerEncoder: the Point-MAE / Point-BERT design (Pang et al. 2022, Yu et al. 2022). Farthest-point
  sampling picks `patches` centres, each centre takes its `patch_size` nearest points, a small PointNet turns each
  patch (positions relative to its centre, plus the other values) into a token, a transformer mixes the tokens with
  a learned embedding of the centre positions, and a class token plus the mean token give the output. With
  `centres="radial"` the centres are picked in radially equalised coordinates (radial_equalise), so they follow the
  points' density: plain farthest-point sampling puts only about 2% of 128 centres in the inner half of a star cloud.
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(dims: list[int], final_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or final_act:
            layers += [nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*layers)


class PointNetEncoder(nn.Module):
    def __init__(self, in_dim: int, widths=(64, 128, 256, 512), out_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.point_mlp = mlp([in_dim, *widths])
        last = widths[-1] if len(widths) else in_dim
        self.head = nn.Sequential(nn.Linear(2 * last, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def make_encoder(kind: str, in_dim: int, out_dim: int, **kw) -> nn.Module:
    if kind == "pointnet":
        return PointNetEncoder(in_dim, kw.get("pointnet_widths", (64, 128, 256, 512)), out_dim, kw.get("dropout", 0.0))
    if kind == "patch":
        return PatchTransformerEncoder(in_dim, kw.get("width", 384), kw.get("depth", 6), kw.get("heads", 6),
                                       kw.get("patches", 128), kw.get("patch_size", 32), kw.get("dropout", 0.0),
                                       out_dim, kw.get("patch_centres", "fps"))
    raise ValueError(f"unknown encoder {kind!r} (pointnet or patch)")
