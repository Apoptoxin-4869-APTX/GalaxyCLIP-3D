"""The multi-modal model: one encoder per modality, each followed by a linear projection into the shared space.

Embeddings are L2-normalized. The temperature is fixed (0.07, as in GalaxyCLIP-2D) unless `learn_temperature`, in
which case it is learned as a log-scale, as in CLIP, capped so the logits never exceed 100.

Ablation options: `columns` ({modality: column indices}) feeds the encoders only those columns of each cloud (e.g.
without velocities), and `shared_encoder` (PointNet only) uses one network for all modalities, each with its own first
layer (the modalities carry different values per point) and its own projection head. `point_invariants` appends to
every point the numbers that do not change when its cloud is rotated or mirrored (`invariant_features`): distance from
the centre, speed, radial velocity and tangential speed, or the distance only for an encoder that does not see
velocities. Positions stay the first three columns, where the patch encoder expects them.

`principal_axes` turns every cloud into its own principal axes before its encoder (`principal_frame`), so the encoders
no longer have to learn from the random rotations that orientation does not matter. The sign of each axis is
arbitrary: in training mode one of the 8 sign choices (`FLIPS`) is drawn per cloud, in eval mode the embeddings of all
8 are averaged, which makes them unchanged by any rotation or mirroring of the input whenever the cloud's three axis
lengths differ (where two are equal, the turn within their plane stays arbitrary). With it, `rotation=shared` and
`independent` give the encoders the same input. It adds no parameters, so checkpoints keep their layout.

The `vn_pointnet` encoder (Vector Neurons) is unchanged by rotations and mirrors by construction, so it takes no
`principal_axes`; it is told per modality whether its input starts with one vector (positions) or two (positions and
velocities), from `columns`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import PointNetEncoder, invariant_features, make_encoder, mlp


# the 8 ways to choose the signs of three axes: 4 rotations and their 4 mirror images
FLIPS = tuple((sx, sy, sz) for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0))


def principal_frame(cloud: torch.Tensor) -> torch.Tensor:
    """(..., N, F) the cloud with its positions (columns 0-2) and velocities (3-5) turned into the principal axes of
    its positions: the eigenvectors of their second-moment tensor about the centre, x along the longest axis and z
    along the shortest. Stars and gas are drawn by mass and dark-matter particles weigh the same, so these are the
    mass-weighted axes. The sign of each axis is arbitrary (see FLIPS); other columns are unchanged, and an all-zero
    (masked) cloud stays all zeros. Computed in float32 and without gradients, since it depends on the data only."""
    with torch.no_grad(), torch.autocast(device_type=cloud.device.type, enabled=False):
        pos = cloud[..., 0:3].float()
        _, axes = torch.linalg.eigh(pos.transpose(-1, -2) @ pos)       # ascending lengths; columns are the axes
        axes = axes.flip(-1)                                            # longest first
        turned = torch.cat([pos @ axes, cloud[..., 3:6].float() @ axes], dim=-1)
    return torch.cat([turned.to(cloud.dtype), cloud[..., 6:]], dim=-1)


def flip_axes(cloud: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """The cloud (..., N, F) with positions and velocities multiplied per axis by signs (..., 3) of +1 or -1."""
    s = torch.cat([signs, signs], dim=-1).to(cloud.dtype).unsqueeze(-2)
    return torch.cat([cloud[..., 0:6] * s, cloud[..., 6:]], dim=-1)


class MultiModalCLIP(nn.Module):
    def __init__(self, feature_dims: dict[str, int], encoder: str = "patch", embed_dim: int = 512,
                 encoder_dim: int = 512, temperature: float = 0.07, learn_temperature: bool = False,
                 columns: dict[str, list[int]] | None = None, shared_encoder: bool = False,
                 point_invariants: bool = False, principal_axes: bool = False, **encoder_kw):
        super().__init__()
        if encoder == "vn_pointnet" and principal_axes:
            raise ValueError("principal_axes does nothing for vn_pointnet, which is rotation-proof by construction")
        self.modalities = list(feature_dims)
        self.principal_axes = bool(principal_axes)
        self.columns = {m: list(columns[m]) for m in feature_dims} if columns is not None else None
        # {modality: whether its invariants include the velocity ones}; None = option off (the model is unchanged)
        self.invariants = ({m: self.columns is None or {3, 4, 5} <= set(self.columns[m]) for m in feature_dims}
                           if point_invariants else None)
        in_dims = {m: (len(self.columns[m]) if self.columns is not None else f)
                   + (0 if self.invariants is None else 4 if self.invariants[m] else 1)
                   for m, f in feature_dims.items()}
        if shared_encoder:
            if encoder != "pointnet":
                raise ValueError("shared_encoder is implemented for the pointnet encoder only")
            widths = tuple(encoder_kw.get("pointnet_widths", (64, 128, 256, 512)))
            trunk = PointNetEncoder(widths[0], widths[1:], encoder_dim, encoder_kw.get("dropout", 0.0),
                                    checkpoint=encoder_kw.get("pointnet_checkpoint", False))
            self.encoders = nn.ModuleDict({m: nn.Sequential(mlp([f, widths[0]]), trunk) for m, f in in_dims.items()})
        else:
            vectors = {m: 1 + (self.columns is None or {3, 4, 5} <= set(self.columns[m])) for m in feature_dims}
            if encoder == "vn_pointnet" and self.columns is not None and any(
                    self.columns[m][:3 * vectors[m]] != list(range(3 * vectors[m])) for m in feature_dims):
                raise ValueError("vn_pointnet needs the positions (and velocities) as the first columns")
            self.encoders = nn.ModuleDict({m: make_encoder(encoder, f, encoder_dim, vectors=vectors[m], **encoder_kw)
                                           for m, f in in_dims.items()})
        self.heads = nn.ModuleDict({m: nn.Linear(encoder_dim, embed_dim) for m in feature_dims})
        scale = torch.tensor(math.log(1.0 / temperature))
        if learn_temperature:
            self.logit_scale = nn.Parameter(scale)
        else:
            self.register_buffer("logit_scale", scale)

    def temperature(self) -> torch.Tensor:
        return 1.0 / self.logit_scale.clamp(max=math.log(100.0)).exp()

    def embed(self, modality: str, cloud: torch.Tensor) -> torch.Tensor:
        if not self.principal_axes:
            return self.embed_as_given(modality, cloud)
        cloud = principal_frame(cloud)
        flips = torch.tensor(FLIPS, device=cloud.device)
        if self.training:                                               # one random sign choice per cloud
            pick = torch.randint(len(FLIPS), cloud.shape[:-2], device=cloud.device)
            return self.embed_as_given(modality, flip_axes(cloud, flips[pick]))
        total = sum(self.embed_as_given(modality, flip_axes(cloud, f.expand(*cloud.shape[:-2], 3))) for f in flips)
        return F.normalize(total, dim=-1)                               # the mean over all 8, normalized again

    def embed_as_given(self, modality: str, cloud: torch.Tensor) -> torch.Tensor:
        """The embedding of the cloud in the orientation it comes in."""
        extra = (invariant_features(cloud, velocities=self.invariants[modality])
                 if self.invariants is not None else None)
        if self.columns is not None:
            cloud = cloud[..., self.columns[modality]]
        if extra is not None:
            cloud = torch.cat([cloud, extra.to(cloud.dtype)], dim=-1)
        return F.normalize(self.heads[modality](self.encoders[modality](cloud)).float(), dim=-1)

    def forward(self, clouds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.embed(m, clouds[m]) for m in self.modalities if m in clouds}
