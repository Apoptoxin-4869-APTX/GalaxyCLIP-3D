"""The multi-modal model: one encoder per modality, each followed by a linear projection into the shared space.

Embeddings are L2-normalized. The temperature is fixed (0.07, as in GalaxyCLIP-2D) unless `learn_temperature`, in
which case it is learned as a log-scale, as in CLIP, capped so the logits never exceed 100.

Ablation options: `columns` ({modality: column indices}) feeds the encoders only those columns of each cloud (e.g.
without velocities), and `shared_encoder` (PointNet only) uses one network for all modalities, each with its own first
layer (the modalities carry different values per point) and its own projection head. `point_invariants` appends to
every point the numbers that do not change when its cloud is rotated or mirrored (`invariant_features`): distance from
the centre, speed, radial velocity and tangential speed, or the distance only for an encoder that does not see
velocities. Positions stay the first three columns, where the patch encoder expects them.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import PointNetEncoder, make_encoder, mlp


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


class MultiModalCLIP(nn.Module):
    def __init__(self, feature_dims: dict[str, int], encoder: str = "patch", embed_dim: int = 512,
                 encoder_dim: int = 512, temperature: float = 0.07, learn_temperature: bool = False,
                 columns: dict[str, list[int]] | None = None, shared_encoder: bool = False,
                 point_invariants: bool = False, **encoder_kw):
        super().__init__()
        self.modalities = list(feature_dims)
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
            trunk = PointNetEncoder(widths[0], widths[1:], encoder_dim, encoder_kw.get("dropout", 0.0))
            self.encoders = nn.ModuleDict({m: nn.Sequential(mlp([f, widths[0]]), trunk) for m, f in in_dims.items()})
        else:
            self.encoders = nn.ModuleDict({m: make_encoder(encoder, f, encoder_dim, **encoder_kw)
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
        extra = (invariant_features(cloud, velocities=self.invariants[modality])
                 if self.invariants is not None else None)
        if self.columns is not None:
            cloud = cloud[..., self.columns[modality]]
        if extra is not None:
            cloud = torch.cat([cloud, extra.to(cloud.dtype)], dim=-1)
        return F.normalize(self.heads[modality](self.encoders[modality](cloud)).float(), dim=-1)

    def forward(self, clouds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.embed(m, clouds[m]) for m in self.modalities if m in clouds}
