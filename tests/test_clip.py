"""Tests of the contrastive training code: orientation, masked loss and recall, encoders, a short training run on a
small synthetic dataset (CPU, a few seconds), and the helpers that score finished runs and fit linear probes. Needs
torch and h5py.

Run: PYTHONPATH=pylibs:. python -s tests/test_clip.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from galaxyclip3d import train_clip                                                    # noqa: E402
from galaxyclip3d.losses import info_nce_loss                                          # noqa: E402
from galaxyclip3d.models import MultiModalCLIP, farthest_point_sample                  # noqa: E402
from galaxyclip3d.models.clip import FLIPS, flip_axes, invariant_features, principal_frame  # noqa: E402
from galaxyclip3d.models.encoders import radial_equalise                               # noqa: E402
from galaxyclip3d.multimodal import MODALITIES, masked_all_pairs_loss, masked_recall, pairs, shuffle_partners  # noqa
from galaxyclip3d.pointclouds.sampling import FEATURES, input_columns, orient          # noqa: E402
from galaxyclip3d.utils import load_checkpoint                                         # noqa: E402

torch.manual_seed(0)
RNG = np.random.default_rng(0)
DIMS = {m: len(FEATURES[m]) for m in MODALITIES}

# ---------------------------------------------------------------- orientation
clouds = {m: RNG.normal(size=(300, d)).astype(np.float32) for m, d in DIMS.items()}
clouds["gas"][:] = 0                                                   # a masked modality stays all zeros
turned = orient(clouds, np.random.default_rng(1), "independent", mirror=True)
for m in MODALITIES:
    a, b = clouds[m], turned[m]
    assert np.allclose(np.linalg.norm(a[:, :3], axis=1), np.linalg.norm(b[:, :3], axis=1), atol=1e-5)
    assert np.allclose((a[:, :3] * a[:, 3:6]).sum(1), (b[:, :3] * b[:, 3:6]).sum(1), atol=1e-4), "r.v kept"
    assert np.array_equal(a[:, 6:], b[:, 6:]), "scalar features untouched"
assert not turned["gas"].any()
fit = {m: np.linalg.lstsq(clouds[m][:, :3], turned[m][:, :3], rcond=None)[0] for m in ("stars", "dm", "hi")}
assert not np.allclose(fit["stars"], fit["dm"], atol=1e-3), "independent: each modality its own matrix"
shared = orient(clouds, np.random.default_rng(1), "shared")
fit = {m: np.linalg.lstsq(clouds[m][:, :3], shared[m][:, :3], rcond=None)[0] for m in ("stars", "dm", "hi")}
assert np.allclose(fit["stars"], fit["dm"], atol=1e-4) and np.allclose(fit["stars"], fit["hi"], atol=1e-4)
assert all(np.array_equal(orient(clouds, None, "none")[m], clouds[m]) for m in MODALITIES)
again = orient(clouds, np.random.default_rng(1), "independent", mirror=True)
assert all(np.array_equal(again[m], turned[m]) for m in MODALITIES), "seeded orientation is reproducible"
dets = [np.linalg.det(np.linalg.lstsq(clouds["stars"][:, :3], orient(clouds, np.random.default_rng(s))["stars"][:, :3],
                                      rcond=None)[0]) for s in range(40)]
assert min(dets) < -0.99 and max(dets) > 0.99, "mirror gives reflections half of the time"
part = orient({m: clouds[m] for m in ("dm", "hi")}, np.random.default_rng(1), "independent", mirror=True)
assert set(part) == {"dm", "hi"} and all(np.array_equal(part[m], turned[m]) for m in part), \
    "a modality gets the same matrix with or without the others (frozen clouds turn alike in every run)"
print("PASS orientation: lengths and r.v kept, scalars untouched, one matrix per modality (or one shared), "
      "reproducible with a seed, reflections with mirror, empty clouds stay empty, a modality turns the same "
      "whichever others are there")

# ---------------------------------------------------------------- masked loss
B, D = 12, 16
emb = {m: torch.nn.functional.normalize(torch.randn(B, D), dim=1) for m in MODALITIES}
full = torch.ones(B, 4, dtype=torch.bool)
loss, per = masked_all_pairs_loss(emb, full, 0.07)
want = sum(info_nce_loss(emb[a], emb[b], 0.07) for a, b in pairs()) / 6
assert torch.allclose(loss, want) and len(per) == 6, "all present: the 2D all-pairs mean"
mask = full.clone()
mask[:5, 2] = False                                                    # five galaxies without gas
loss1, per1 = masked_all_pairs_loss(emb, mask, 0.07)
emb2 = dict(emb)
emb2["gas"] = emb["gas"].clone()
emb2["gas"][:5] = torch.nn.functional.normalize(torch.randn(5, D), dim=1)
loss2, _ = masked_all_pairs_loss(emb2, mask, 0.07)
assert torch.allclose(loss1, loss2), "masked galaxies do not affect the loss"
rows = ~torch.zeros(B, dtype=torch.bool)
rows[:5] = False
assert abs(per1["stars-gas"] - float(info_nce_loss(emb["stars"][rows], emb["gas"][rows], 0.07))) < 1e-6
mask[:, 3] = False
mask[0, 3] = True                                                      # HI in one galaxy only: its pairs skipped
_, per3 = masked_all_pairs_loss(emb, mask, 0.07)
assert set(per3) == {"stars-dm", "stars-gas", "dm-gas"}
anchor = pairs(anchor="stars")
assert anchor == [("stars", "dm"), ("stars", "gas"), ("stars", "hi")]
la, pa = masked_all_pairs_loss(emb, full, 0.07, pair_list=anchor)
assert set(pa) == {"stars-dm", "stars-gas", "stars-hi"}
assert torch.allclose(la, sum(info_nce_loss(emb[a], emb[b], 0.07) for a, b in anchor) / 3), "stars anchor: 3 pairs"
x = {m: torch.randn(B, D, requires_grad=True) for m in MODALITIES}
l, _ = masked_all_pairs_loss({m: torch.nn.functional.normalize(v, dim=1) for m, v in x.items()}, full, 0.07)
l.backward()
assert all(x[m].grad is not None and x[m].grad.abs().sum() > 0 for m in MODALITIES)
sh, shm = shuffle_partners(emb, full, torch.Generator().manual_seed(0))
assert torch.equal(sh["stars"], emb["stars"]) and not torch.equal(sh["dm"], emb["dm"])
print("PASS masked loss: equals the 2D all-pairs loss when nothing is missing, ignores masked galaxies, skips pairs "
      "without enough galaxies, gradients reach every modality; the shuffle control re-pairs galaxies; a stars anchor "
      "trains only the three stars pairs")

# ---------------------------------------------------------------- masked recall
N = 200
base = torch.nn.functional.normalize(torch.randn(N, 32), dim=1)
perfect = {m: base for m in MODALITIES}
m2 = torch.ones(N, 4, dtype=torch.bool)
m2[:50, 2] = False
r = masked_recall(perfect, m2)
assert r["r1/mean"] == 1.0 and r["gallery/stars->gas"] == 150 and r["gallery/stars->dm"] == 200
assert abs(r["chance/stars->gas"] - 1 / 150) < 1e-9
rand = {m: torch.nn.functional.normalize(torch.randn(N, 32), dim=1) for m in MODALITIES}
r = masked_recall(rand, m2)
assert r["r1/mean"] < 0.05 and r["r10/mean"] < 0.15, r["r1/mean"]
same = {m: torch.ones(N, 8) / 8 ** 0.5 for m in MODALITIES}
assert masked_recall(same, m2)["r1/mean"] == 0.0, "ties count against: a collapsed model scores zero"
groups = torch.arange(N) % 4
r = masked_recall(rand, m2, groups=groups)
assert abs(r["gallery/stars->dm"] - 50) < 1e-9 and abs(r["chance/stars->dm"] - 0.02) < 1e-9
gh = masked_recall(perfect, m2)
assert "r1/gas->hi" in gh and gh["r1/mean_no_gas_hi"] == 1.0
print("PASS masked recall: perfect embeddings 100%, random near chance, ties count against, galleries follow the "
      "masks, mass groups shrink the gallery, gas-HI kept out of the second mean")

# ---------------------------------------------------------------- encoders
for kind, extra in (("pointnet", {}), ("patch", {}), ("patch", {"patch_centres": "radial"}),
                    ("pointnet", {"point_invariants": True}), ("patch", {"point_invariants": True}),
                    ("pointnet2", {"sa_centres": (16, 4), "sa_neighbours": (8, 4), "sa_widths": ((16, 16), (16, 32))}),
                    ("pointnet2", {"sa_centres": (16, 4), "sa_neighbours": (8, 4), "sa_widths": ((16, 16), (16, 32)), "sa_sampling": "fps", "point_invariants": True}),
                    ("pointnext", {"nx_width": 8, "nx_centres": (32, 16, 8, 4), "nx_neighbours": (8, 8, 4, 4), "nx_widths": (16, 16, 32, 32), "nx_blocks": (1, 2, 0, 1)}),
                    ("pointnext", {"nx_width": 8, "nx_centres": (32, 16, 8, 4), "nx_neighbours": (8, 8, 4, 4), "nx_widths": (16, 16, 32, 32), "nx_blocks": (1, 2, 0, 1), "sa_sampling": "fps", "point_invariants": True}),
                    ("vn_pointnet", {"vn_widths": (8, 16), "vn_scalar_widths": (16, 32)}),
                    ("vn_pointnet", {"vn_widths": (8, 16), "vn_scalar_widths": (16, 32), "point_invariants": True})):
    model = MultiModalCLIP(DIMS, kind, embed_dim=64, encoder_dim=64, width=64, depth=2, heads=4, patches=16,
                           patch_size=8, pointnet_widths=(16, 32), **extra)
    xb = {m: torch.randn(3, 128, d) for m, d in DIMS.items()}
    xb["gas"][1] = 0
    out = model(xb)
    perm = torch.randperm(128)
    out2 = model({m: v[:, perm] for m, v in xb.items()})
    assert all(out[m].shape == (3, 64) and torch.allclose(out[m].norm(dim=1), torch.ones(3)) for m in MODALITIES)
    assert all(torch.allclose(out[m], out2[m], atol=1e-5) for m in MODALITIES), f"{kind}: point order matters"
    assert torch.isfinite(out["gas"]).all(), "an all-zero (masked) cloud must not give NaN"
idx = farthest_point_sample(torch.randn(2, 100, 3), 10)
assert idx.shape == (2, 10) and all(len(set(i.tolist())) == 10 for i in idx)
from galaxyclip3d.models.encoders import PointNet2Encoder                             # noqa: E402
pn2 = PointNet2Encoder(DIMS["gas"], 32, centres=(16, 4), neighbours=(8, 4), widths=((16,), (16, 32)), global_widths=(32,))
assert pn2(torch.randn(2, 64, DIMS["gas"])).shape == (2, 32)
assert pn2(torch.randn(2, 6, DIMS["gas"])).shape == (2, 32), "fewer points than centres or neighbours still works"
assert pn2.levels[0][0].in_features == 3 + 2 * DIMS["gas"] and pn2.levels[1][0].in_features == 3 + 2 * (2 * 16 + 4)
for bad in (dict(sampling="random"), dict(centres=(16, 32)), dict(centres=(16,)), dict(neighbours=(0, 4))):
    try:
        PointNet2Encoder(8, 32, **{"centres": (16, 4), "neighbours": (8, 4), "widths": ((16,), (16,)), **bad})
        raise AssertionError(f"{bad} should be refused")
    except ValueError:
        pass
from galaxyclip3d.models.encoders import InvResMLP, PointNeXtEncoder, SetAbstraction  # noqa: E402
nx = PointNeXtEncoder(DIMS["gas"], 32, width=8, centres=(32, 16, 8), neighbours=(8, 8, 4), widths=(16, 16, 32),
                      blocks=(1, 0, 2))
assert nx(torch.randn(2, 64, DIMS["gas"])).shape == (2, 32) and nx(torch.randn(2, 5, DIMS["gas"])).shape == (2, 32)
kinds = [[type(b).__name__ for b in st] for st in nx.stages]
assert kinds == [["SetAbstraction", "InvResMLP"], ["SetAbstraction"], ["SetAbstraction", "InvResMLP", "InvResMLP"]]
assert isinstance(nx.stages[0][0].mlp, torch.nn.Linear) and isinstance(nx.stages[1][0].mlp, torch.nn.Sequential), \
    "one layer per point in a set abstraction followed by blocks, two without (as in PointNeXt)"
assert nx.stages[2][1].pointwise[0].out_features == 4 * 32, "blocks widen 4x"
blk, xyz0, f0 = InvResMLP(16, 4), torch.randn(2, 10, 3), torch.randn(2, 10, 16)
torch.nn.init.zeros_(blk.pointwise[3].weight), torch.nn.init.zeros_(blk.pointwise[3].bias)
torch.nn.init.zeros_(blk.pointwise[4].weight), torch.nn.init.zeros_(blk.pointwise[4].bias)
assert torch.allclose(blk(xyz0, f0), torch.nn.functional.gelu(f0)), "a block whose last layer is zero passes its input on"
for bad in (dict(sampling="random"), dict(centres=(16, 32, 8)), dict(blocks=(1, 1)), dict(blocks=(1, -1, 1)),
            dict(expansion=0)):
    try:
        PointNeXtEncoder(8, 32, **{"width": 8, "centres": (16, 8, 4), "neighbours": (4, 4, 4), "widths": (8, 8, 8),
                                   "blocks": (1, 1, 1), **bad})
        raise AssertionError(f"{bad} should be refused")
    except ValueError:
        pass
lt = MultiModalCLIP(DIMS, "pointnet", embed_dim=8, encoder_dim=8, learn_temperature=True, pointnet_widths=(8,))
assert isinstance(lt.logit_scale, torch.nn.Parameter) and abs(float(lt.temperature().detach()) - 0.07) < 1e-6
print("PASS encoders (PointNet, patch transformer, PointNet++, PointNeXt, Vector Neurons): shapes, unit-length "
      "embeddings, same "
      "output for shuffled points, masked clouds finite, PointNet++ levels sized as designed and bad levels refused, "
      "PointNeXt stages as designed (one or two layers per set abstraction, blocks widened 4x and residual), bad "
      "stages refused, "
      "farthest-point sampling picks distinct points, learnable temperature")

# ---------------------------------------------------------------- ablation options of the model
xb = {m: torch.randn(2, 64, d) for m, d in DIMS.items()}
for inputs, cols in (("no_velocities", slice(3, 6)), ("no_scalars", slice(6, None))):
    mdl = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16,),
                         columns={m: input_columns(m, inputs) for m in MODALITIES})
    changed = {m: v.clone() for m, v in xb.items()}
    for v in changed.values():
        v[..., cols] += 5.0
    a, b = mdl(xb), mdl(changed)
    assert all(torch.allclose(a[m], b[m]) for m in MODALITIES), f"{inputs}: dropped columns must not matter"
    assert mdl.encoders["dm"].point_mlp[0].in_features == len(input_columns("dm", inputs))
plain = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16,))
changed = {m: v.clone() for m, v in xb.items()}
changed["stars"][..., 3:6] += 5.0
assert not torch.allclose(plain(xb)["stars"], plain(changed)["stars"]), "with all columns, velocities matter"
assert input_columns("dm", "no_velocities") == [0, 1, 2] and input_columns("gas", "no_scalars") == list(range(6))
no_age = train_clip.TrainConfig(modalities=["stars", "dm", "gas"], drop_inputs=["stars.age"], encoder="pointnet",
                                 pointnet_widths=[16], embed_dim=16, encoder_dim=16)
cols = train_clip.encoder_columns(no_age)
assert cols["stars"] == [0, 1, 2, 3, 4, 5, 7] and cols["dm"] == list(range(6)) and cols["gas"] == list(range(12))
assert train_clip.encoder_columns(train_clip.TrainConfig()) is None, "nothing hidden: clouds go in as they are"
age = FEATURES["stars"].index("age")
for kind in ("pointnet", "patch"):
    cfg_k = train_clip.TrainConfig(modalities=["stars", "dm", "gas"], drop_inputs=["stars.age"], encoder=kind,
                                   pointnet_widths=[16], embed_dim=16, encoder_dim=16, width=16, depth=1, heads=2,
                                   patches=8, patch_size=8)
    mdl = train_clip.build_model(cfg_k).eval()
    first = mdl.encoders["stars"].point_mlp[0] if kind == "pointnet" else mdl.encoders["stars"].local1[0]
    assert first.in_features == len(FEATURES["stars"]) - 1, f"{kind}: one stellar input fewer"
    sub = {m: xb[m][..., :len(FEATURES[m])] for m in cfg_k.modalities}
    older = {m: v.clone() for m, v in sub.items()}
    older["stars"][..., age] += 5.0
    richer = {m: v.clone() for m, v in sub.items()}
    richer["stars"][..., FEATURES["stars"].index("log_z")] += 5.0
    with torch.no_grad():
        a, b, c = mdl(sub), mdl(older), mdl(richer)
    assert torch.allclose(a["stars"], b["stars"]), f"{kind}: the hidden age must not matter"
    assert not torch.allclose(a["stars"], c["stars"]), f"{kind}: metallicity still matters"
sh = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16, 32), shared_encoder=True)
sep = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16, 32))
assert sh.encoders["stars"][1] is sh.encoders["dm"][1] is sh.encoders["hi"][1], "one trunk for all modalities"
assert sh.encoders["stars"][0][0].in_features == DIMS["stars"] and sh.encoders["gas"][0][0].in_features == DIMS["gas"]
assert sum(p.numel() for p in sh.parameters()) < sum(p.numel() for p in sep.parameters())
perm = torch.randperm(64)
o1, o2 = sh(xb), sh({m: v[:, perm] for m, v in xb.items()})
assert all(o1[m].shape == (2, 16) and torch.allclose(o1[m], o2[m], atol=1e-5) for m in MODALITIES)
sh2 = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16, 32), shared_encoder=True)
sh2.load_state_dict(sh.state_dict())
assert sh2.encoders["stars"][1] is sh2.encoders["gas"][1] and all(torch.allclose(sh2(xb)[m], o1[m]) for m in MODALITIES)
try:
    MultiModalCLIP(DIMS, "patch", shared_encoder=True)
    raise AssertionError("a shared patch encoder should be refused")
except ValueError:
    pass
dirs = torch.nn.functional.normalize(torch.randn(2, 2048, 3), dim=-1)
rad = torch.distributions.Exponential(torch.tensor(8.0)).sample((2, 2048)).clamp(max=1.0)   # dense core, thin halo
cloud = dirs * rad[..., None]
half = rad.median(dim=1, keepdim=True).values
inner = {name: float((torch.gather(rad, 1, farthest_point_sample(pts, 128)) < half).float().mean())
         for name, pts in (("fps", cloud), ("radial", radial_equalise(cloud)))}
assert inner["radial"] > 0.3 and inner["fps"] < 0.15, inner
assert torch.allclose(torch.nn.functional.normalize(radial_equalise(cloud), dim=-1), dirs, atol=1e-5)
pts = torch.randn(2, 50, 8)
rot = torch.linalg.qr(torch.randn(3, 3))[0]
rot = rot * torch.det(rot).sign()                                      # a proper rotation
for mat in (rot, -rot):                                                # -rot also mirrors
    moved = pts.clone()
    moved[..., 0:3], moved[..., 3:6] = pts[..., 0:3] @ mat.T, pts[..., 3:6] @ mat.T
    assert torch.allclose(invariant_features(pts), invariant_features(moved), atol=1e-5), "rotation or mirror changed it"
one = torch.tensor([[[2.0, 0.0, 0.0, 1.0, 1.0, 0.0]]])                 # r 2, speed sqrt 2, radial 1, tangential 1
assert torch.allclose(invariant_features(one), torch.tensor([[[2.0, 2 ** 0.5, 1.0, 1.0]]]))
assert torch.allclose(invariant_features(one, velocities=False), torch.tensor([[[2.0]]]))
zero = invariant_features(torch.zeros(1, 5, 6))
assert torch.isfinite(zero).all() and not zero.any(), "a masked cloud gives zeros"
inv = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16,), point_invariants=True)
assert all(inv.encoders[m].point_mlp[0].in_features == DIMS[m] + 4 for m in MODALITIES)
for inputs, added in (("no_velocities", 1), ("no_scalars", 4)):
    mdl = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16,), point_invariants=True,
                         columns={m: input_columns(m, inputs) for m in MODALITIES})
    assert all(mdl.encoders[m].point_mlp[0].in_features == len(input_columns(m, inputs)) + added for m in MODALITIES)
nv = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16,), point_invariants=True,
                    columns={m: input_columns(m, "no_velocities") for m in MODALITIES})
changed = {m: v.clone() for m, v in xb.items()}
for v in changed.values():
    v[..., 3:6] += 5.0
assert all(torch.allclose(nv(xb)[m], nv(changed)[m]) for m in MODALITIES), "velocities leaked in through the invariants"
assert not torch.allclose(inv(xb)["stars"], inv(changed)["stars"]), "with velocities, the invariants use them"
print(f"PASS ablation options: dropped input columns are ignored (a hidden stellar age too, PointNet and patch), "
      f"one shared PointNet trunk with its own first layer "
      f"per modality (fewer parameters, checkpoint round trip), radial patch centres put {inner['radial']:.0%} of "
      f"centres in the inner half of a concentrated cloud against {inner['fps']:.0%} for plain farthest-point sampling; "
      f"point invariants are unchanged by rotation and mirroring, exact on a known point, zero for masked clouds, and "
      f"carry no velocities when velocities are dropped")


# ---------------------------------------------------------------- principal axes
gen = torch.Generator().manual_seed(3)
LENGTHS = torch.tensor([3.0, 2.0, 1.0])                                # distinct axis lengths: a well-defined frame


def o3(g):
    """A random 3 x 3 orthogonal matrix: a rotation, or a rotation and a mirror."""
    return torch.linalg.qr(torch.randn(3, 3, generator=g))[0]


def turn(cloud, mat):
    out = cloud.clone()
    out[..., 0:3], out[..., 3:6] = cloud[..., 0:3] @ mat.T, cloud[..., 3:6] @ mat.T
    return out


base = torch.randn(4, 500, 8, generator=gen)
base[..., 0:3] *= LENGTHS
pf = principal_frame(base)
mom = pf[..., 0:3].transpose(-1, -2) @ pf[..., 0:3]
diag = torch.diagonal(mom, dim1=-2, dim2=-1)
assert (mom - torch.diag_embed(diag)).abs().max() < 1e-4 * diag.max(), "second moments are diagonal in the frame"
assert (diag[:, 0] > diag[:, 1]).all() and (diag[:, 1] > diag[:, 2]).all(), "x along the longest axis, z the shortest"
assert torch.equal(pf[..., 6:], base[..., 6:]), "scalar columns untouched"
assert torch.allclose(invariant_features(pf), invariant_features(base), atol=1e-4), "a rotation, not a distortion"
mirrored = 0
for mat in [o3(gen) for _ in range(3)]:
    for m3 in (mat, -mat):                                             # opposite determinants: one of them mirrors
        mirrored += int(torch.det(m3) < 0)
        moved = principal_frame(turn(base, m3))
        signs = torch.sign((moved[..., 0:3] * pf[..., 0:3]).sum(-2))  # (clouds, 3): how each axis came out
        assert torch.allclose(moved, flip_axes(pf, signs), atol=1e-4), "same frame up to the sign of each axis"
assert mirrored == 3
zero = principal_frame(torch.zeros(2, 10, 7))
assert torch.isfinite(zero).all() and not zero.any(), "a masked cloud stays all zeros"
assert len(set(FLIPS)) == 8 and sum(int(np.prod(f) < 0) for f in FLIPS) == 4, "4 rotations, 4 mirror images"

pa = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16, 32), principal_axes=True)
asis = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(16, 32))
asis.load_state_dict(pa.state_dict())                                 # no extra parameters: same layout
xb = {m: torch.randn(3, 200, d, generator=gen) for m, d in DIMS.items()}
for v in xb.values():
    v[..., 0:3] *= LENGTHS
xb["gas"][1] = 0                                                       # a masked cloud
pa.eval(), asis.eval()
with torch.no_grad():
    e0, a0 = pa(xb), asis(xb)
    for _ in range(3):                                                 # every modality turned on its own
        moved = {m: turn(v, o3(gen)) for m, v in xb.items()}
        e1, a1 = pa(moved), asis(moved)
        assert all(torch.allclose(e0[m], e1[m], atol=1e-5) for m in MODALITIES), "eval: rotation or mirror changed it"
        assert not torch.allclose(a0["stars"], a1["stars"], atol=1e-3), "without the frame, orientation matters"
    assert all(torch.isfinite(e0[m]).all() and torch.allclose(e0[m].norm(dim=1), torch.ones(3)) for m in MODALITIES)
    pa.train()
    torch.manual_seed(5)
    picks = []
    for _ in range(4):
        et = pa.embed("stars", xb["stars"])
        frame = principal_frame(xb["stars"])
        each = torch.stack([pa.embed_as_given("stars", flip_axes(frame, torch.tensor(f).expand(3, 3))) for f in FLIPS])
        hit = (each - et[None]).abs().amax(-1) < 1e-5                  # (8 sign choices, 3 clouds)
        assert (hit.sum(0) >= 1).all(), "training: each cloud gets one of the 8 sign choices"
        picks += hit.float().argmax(0).tolist()
    assert len(set(picks)) > 2, "training: the sign choices vary"
print(f"PASS principal axes: the frame diagonalises the second moments (longest axis first), leaves scalars alone, is "
      f"a pure rotation, and comes out the same up to axis signs after 6 random rotations or mirrorings ({mirrored} "
      f"mirrored); masked clouds stay zeros; in eval mode the model's embeddings are unchanged when every modality is "
      f"turned on its own (the plain model's are not), with no extra parameters; in training mode each cloud gets one "
      f"of the 8 sign choices, {len(set(picks))} different ones over 12 draws")

# ---------------------------------------------------------------- Vector Neurons
from galaxyclip3d.models.encoders import VNBlock, VNPointNetEncoder, vn_leaky_relu, vn_rms  # noqa: E402
gen = torch.Generator().manual_seed(4)
v0, s0, d0 = (torch.randn(2, 50, 3, 4, generator=gen), torch.randn(2, 50, 5, generator=gen),
              torch.randn(2, 50, 3, 4, generator=gen))
blk = VNBlock(4, 5, 6, 7)
mirrored = 0
for mat in [o3(gen) for _ in range(3)]:
    for m3 in (mat, -mat):                                             # opposite determinants: one of them mirrors
        mirrored += int(torch.det(m3) < 0)

        def rot(v, m3=m3):
            return torch.einsum("ij,bnjc->bnic", m3, v)
        vb, sb = blk(v0, s0)
        vr, sr = blk(rot(v0), s0)
        assert torch.allclose(vr, rot(vb), atol=1e-5) and torch.allclose(sr, sb, atol=1e-5), \
            "a block commutes with turns"
        assert torch.allclose(vn_rms(rot(v0)), vn_rms(v0), atol=1e-5), "the root-mean-square length is invariant"
        assert torch.allclose(vn_leaky_relu(rot(v0), rot(d0)), rot(vn_leaky_relu(v0, d0)), atol=1e-5)
assert mirrored == 3
along, d1 = torch.tensor([[2.0], [1.0], [0.0]]), torch.tensor([[1.0], [0.0], [0.0]])     # (3, 1): one channel
assert torch.equal(vn_leaky_relu(along, d1), along), "a vector on its direction's side passes unchanged"
assert torch.allclose(vn_leaky_relu(-along * torch.tensor([[1.0], [-1.0], [1.0]]), d1, 0.2),
                      torch.tensor([[-0.4], [1.0], [0.0]]), atol=0.02), \
    "the opposite side keeps 20% of its part along d"
assert torch.allclose(vn_leaky_relu(2.5 * v0, d0), 2.5 * vn_leaky_relu(v0, d0), atol=1e-5), \
    "positive factors pass through the LeakyReLU (so the block divides by the RMS length after it)"
vz, sz = blk(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 5))
assert torch.equal(vz, torch.zeros(1, 2, 3, 6)) and torch.isfinite(sz).all(), "a masked cloud's vectors stay zeros"

vn = MultiModalCLIP(DIMS, "vn_pointnet", embed_dim=16, encoder_dim=16, vn_widths=(8, 16, 16),
                    vn_scalar_widths=(16, 32, 32))
xb = {m: torch.randn(3, 200, d, generator=gen) for m, d in DIMS.items()}
xb["gas"][1] = 0                                                       # a masked cloud
for mode in ("train", "eval"):                                         # no frame, no averaging: the same in both modes
    getattr(vn, mode)()
    with torch.no_grad():
        e0 = vn(xb)
        for _ in range(2):
            for sign in (1, -1):                                       # every modality turned on its own
                e1 = vn({m: turn(v, sign * o3(gen)) for m, v in xb.items()})
                assert all(torch.allclose(e0[m], e1[m], atol=1e-5) for m in MODALITIES), f"{mode}: a turn changed it"
    assert all(torch.isfinite(e0[m]).all() and torch.allclose(e0[m].norm(dim=1), torch.ones(3)) for m in MODALITIES)
with torch.no_grad():
    for col, what in ((slice(6, None), "scalars"), (slice(3, 6), "velocities")):
        changed = {m: v.clone() for m, v in xb.items()}
        changed["stars"][..., col] += 0.5
        assert not torch.allclose(vn(changed)["stars"], e0["stars"], atol=1e-3), f"the stars' {what} are used"
for inputs, n_vec in (("no_velocities", 1), ("no_scalars", 2)):
    cols = {m: input_columns(m, inputs) for m in MODALITIES}
    sub = MultiModalCLIP(DIMS, "vn_pointnet", embed_dim=16, encoder_dim=16, vn_widths=(8, 16),
                         vn_scalar_widths=(16, 32), columns=cols).eval()
    assert all(sub.encoders[m].blocks[0].mix.in_features == n_vec for m in MODALITIES), inputs
    with torch.no_grad():
        e0s, e1s = sub(xb), sub({m: turn(v, -o3(gen)) for m, v in xb.items()})
    assert all(torch.allclose(e0s[m], e1s[m], atol=1e-5) for m in MODALITIES), f"{inputs}: a turn changed it"
vn.train()
mk = torch.ones(3, len(MODALITIES), dtype=torch.bool)
mk[1, MODALITIES.index("gas")] = False
loss, _ = masked_all_pairs_loss(vn(xb), mk, vn.temperature())
loss.backward()
assert all(torch.isfinite(q.grad).all() for q in vn.parameters() if q.grad is not None), "finite gradients"
assert float(vn.encoders["dm"].pool_direction.weight.grad.abs().max()) > 0, "the pooling direction is trained"
ckpt = MultiModalCLIP(DIMS, "vn_pointnet", embed_dim=16, encoder_dim=16, vn_widths=(8, 16, 16),
                      vn_scalar_widths=(16, 32, 32), vn_checkpoint=True).train()
ckpt.load_state_dict(vn.state_dict())
assert all(e.checkpoint for e in ckpt.encoders.values()) and not any(e.checkpoint for e in vn.encoders.values())
loss_c, _ = masked_all_pairs_loss(ckpt(xb), mk, ckpt.temperature())
loss_c.backward()
same_grads = all(torch.allclose(a.grad, b.grad, atol=1e-6)
                 for a, b in zip(ckpt.parameters(), vn.parameters()) if a.grad is not None)
assert torch.allclose(loss_c, loss, atol=1e-6) and same_grads, \
    "recomputing the layers in the backward pass gives the same loss and gradients"
for bad in (lambda: MultiModalCLIP(DIMS, "vn_pointnet", principal_axes=True),
            lambda: MultiModalCLIP(DIMS, "vn_pointnet", columns={m: [3, 4, 5, 0, 1, 2] for m in MODALITIES}),
            lambda: VNPointNetEncoder(8, widths=(8,), scalar_widths=(8, 16)), lambda: VNPointNetEncoder(8, vectors=3),
            lambda: VNPointNetEncoder(4, vectors=2)):
    try:
        bad()
        raise AssertionError("should be refused")
    except ValueError:
        pass
print(f"PASS Vector Neurons: a block and the LeakyReLU commute with 6 random turns ({mirrored} mirrored) and the "
      f"RMS length is invariant; the LeakyReLU passes vectors on its direction's side, keeps 20% of the rest and lets "
      f"positive factors through; masked clouds stay zeros; the model's embeddings are unchanged when every "
      f"modality is turned on its own, in training and eval mode and without velocities or scalars; scalars and "
      f"velocities are used; gradients are finite and reach the pooling direction; checkpointing gives the same loss "
      f"and gradients; principal axes, misplaced vector columns and bad widths are refused")

for shared in (False, True):                       # pointnet_checkpoint: layers recomputed in the backward pass
    torch.manual_seed(5)
    plain = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(8, 16, 32),
                           shared_encoder=shared).train()
    redo = MultiModalCLIP(DIMS, "pointnet", embed_dim=16, encoder_dim=16, pointnet_widths=(8, 16, 32),
                          shared_encoder=shared, pointnet_checkpoint=True).train()
    redo.load_state_dict(plain.state_dict())                            # the option changes no weights
    assert all((e[1] if shared else e).checkpoint for e in redo.encoders.values()), "it reaches every PointNet"
    assert not any((e[1] if shared else e).checkpoint for e in plain.encoders.values())
    pn_losses = []
    for model in (plain, redo):
        loss_m, _ = masked_all_pairs_loss(model(xb), mk, model.temperature())
        loss_m.backward()
        pn_losses.append(loss_m.detach())
    assert torch.equal(pn_losses[0], pn_losses[1]), f"shared={shared}: the same loss"
    assert all(torch.allclose(a.grad, b.grad, rtol=0, atol=1e-7) for a, b in zip(plain.parameters(), redo.parameters())
               if a.grad is not None), f"shared={shared}: the same gradients"
    with torch.no_grad():
        e_plain, e_redo = plain.eval()(xb), redo.eval()(xb)
    assert all(torch.equal(e_plain[m], e_redo[m]) for m in MODALITIES), f"shared={shared}: the same embeddings"
print("PASS PointNet checkpointing: recomputing the layers in the backward pass gives the same loss, gradients and "
      "embeddings and the same weights, with one encoder per modality and with a shared one")


# ---------------------------------------------------------------- a short training run on synthetic galaxies
class Synthetic(torch.utils.data.Dataset):
    """Galaxies whose four clouds share a hidden shape (axis lengths) and scalar values; each cloud is rotated on
    its own, so only intrinsic properties can match them."""

    def __init__(self, n, seed, frozen):
        g = np.random.default_rng(seed)
        self.axes = g.uniform(0.2, 1.0, size=(n, 3))
        self.level = g.normal(size=(n, 1))
        self.frozen, self.n = frozen, n
        self.mask = np.ones((n, 4), dtype=bool)
        self.mask[: n // 4, 2] = False

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = np.random.default_rng(i if self.frozen else None)
        c = {}
        for m, d in DIMS.items():
            pts = g.normal(size=(64, d)).astype(np.float32)
            pts[:, :3] *= self.axes[i]
            pts[:, 6:] += self.level[i]
            c[m] = pts if self.mask[i, MODALITIES.index(m)] else np.zeros_like(pts)
        c = orient(c, g, "independent")
        item = {m: torch.from_numpy(v) for m, v in c.items()}
        item.update(mask=torch.from_numpy(self.mask[i].copy()), mstar=float(1e9 * (1 + i % 3)))
        return item


# ---------------------------------------------------------------- training and validation snapshots
seen = []
real_dataset = train_clip.PointCloudDataset
train_clip.PointCloudDataset = lambda root, split, **kw: seen.append((split, kw["snapshots"]))
for over, want in (([], [None, None]), (["snapshots=[91,99]"], [[91, 99], [91, 99]]),
                   (["val_snapshots=[99]"], [None, [99]]),
                   (["snapshots=[50,99]", "val_snapshots=[99]"], [[50, 99], [99]])):
    seen.clear()
    train_clip.build_datasets(train_clip.TrainConfig.from_file(None, over))
    assert seen == [("train", want[0]), ("val", want[1])], (over, seen)
train_clip.PointCloudDataset = real_dataset
try:
    train_clip.TrainConfig.from_file(None, ["val_snapshots=[]"])
    raise AssertionError("an empty val_snapshots should be refused")
except SystemExit:
    pass
print("PASS snapshots: training uses `snapshots` and validation `val_snapshots` (by default the same); an empty "
      "val_snapshots is refused")

tmp = Path(tempfile.mkdtemp(prefix="clip_test_"))
train_clip.build_datasets = lambda cfg: (Synthetic(96, 1, False), Synthetic(40, 2, True))
common = ["encoder=\"pointnet\"", "pointnet_widths=[32,64]", "encoder_dim=64", "embed_dim=32", "batch_size=16",
          "workers=0", "eval_every=1", "warmup_epochs=1", "lr=3e-3", f"out_dir=\"{tmp / 'run'}\"", "amp=\"none\""]
train_clip.main(["--set", *common, "epochs=3"])
hist = json.loads((tmp / "run" / "history.json").read_text())
assert [v["epoch"] for v in hist["val"]] == [0, 1, 2, 3] and len(hist["train_loss"]) == 3
assert (tmp / "run" / "best.pt").exists() and (tmp / "run" / "latest.pt").exists()
assert all(np.isfinite(hist["train_loss"])) and hist["train_loss"][-1] < hist["train_loss"][0]
train_clip.main(["--set", *common, "epochs=4"])
hist = json.loads((tmp / "run" / "history.json").read_text())
assert [v["epoch"] for v in hist["val"]] == [0, 1, 2, 3, 4] and len(hist["train_loss"]) == 4, "resumed at epoch 3"
train_clip.main(["--set", *common, "epochs=2", "control=\"shuffle_pairs\"", f"out_dir=\"{tmp / 'shuffled'}\""])
train_clip.main(["--set", *common, "epochs=1", "inputs=\"no_velocities\"", "loss_pairs=\"stars_anchor\"",
                 "shared_encoder=true", "ema_decay=0.9", "point_invariants=true", 'drop_inputs=["stars.age"]',
                 f"out_dir=\"{tmp / 'ablation'}\""])
ab = json.loads((tmp / "ablation" / "history.json").read_text())
assert set(ab["pair_loss"][0]) == {"stars-dm", "stars-gas", "stars-hi"}, "the anchor trains only the stars pairs"
assert "r1/dm->gas" in ab["val"][-1] and len(ab["val"]) == 2, "validation keeps every direction"
train_clip.main(["--set", *common, "epochs=1", 'encoder="pointnet2"', "sa_centres=[16,4]", "sa_neighbours=[8,4]",
                 "sa_widths=[[16,16],[16,32]]", f"out_dir=\"{tmp / 'pn2'}\""])
pn = json.loads((tmp / "pn2" / "history.json").read_text())
assert len(pn["val"]) == 2 and all(np.isfinite(pn["train_loss"])), "a PointNet++ run trains and validates"
pn_cfg = train_clip.TrainConfig(**json.loads((tmp / "pn2" / "config.json").read_text()))
pn_model = train_clip.build_model(pn_cfg)
pn_model.load_state_dict(load_checkpoint(tmp / "pn2" / "best.pt")["model_state_dict"])
assert type(pn_model.encoders["stars"]).__name__ == "PointNet2Encoder" and len(pn_model.encoders["stars"].levels) == 2
train_clip.main(["--set", *common, "epochs=1", 'encoder="pointnext"', "nx_width=8", "nx_centres=[32,16,8,4]",
                 "nx_neighbours=[8,8,4,4]", "nx_widths=[16,16,32,32]", "nx_blocks=[1,1,0,1]",
                 f"out_dir=\"{tmp / 'pnx'}\""])
px = json.loads((tmp / "pnx" / "history.json").read_text())
assert len(px["val"]) == 2 and all(np.isfinite(px["train_loss"])), "a PointNeXt run trains and validates"
px_cfg = train_clip.TrainConfig(**json.loads((tmp / "pnx" / "config.json").read_text()))
px_model = train_clip.build_model(px_cfg)
px_model.load_state_dict(load_checkpoint(tmp / "pnx" / "best.pt")["model_state_dict"])
assert type(px_model.encoders["gas"]).__name__ == "PointNeXtEncoder" and len(px_model.encoders["gas"].stages) == 4
for bad in (['encoder="pointnet3"'], ['sa_sampling="random"'], ['encoder="pointnet2"', "sa_centres=[16,32]"],
            ['encoder="pointnet2"', "sa_neighbours=[8]"], ['encoder="pointnext"', "nx_blocks=[1,1]"],
            ['encoder="pointnext"', "nx_centres=[8,16,4,2]"], ['encoder="pointnext"', "nx_expansion=0"],
            ['encoder="vn_pointnet"', "vn_widths=[8]"], ['encoder="vn_pointnet"', "vn_widths=[0,8,8,8]"],
            ['encoder="vn_pointnet"', "principal_axes=true"], ['encoder="vn_pointnet"', "pointnet_checkpoint=true"]):
    try:
        train_clip.TrainConfig.from_file(None, bad)
        raise AssertionError(f"{bad} should be refused")
    except SystemExit:
        pass
train_clip.main(["--set", *common, "epochs=1", "principal_axes=true", f"out_dir=\"{tmp / 'principal'}\""])
pr = json.loads((tmp / "principal" / "history.json").read_text())
assert len(pr["val"]) == 2 and all(np.isfinite(pr["train_loss"])), "a principal-axes run trains and validates"
pr_dict = json.loads((tmp / "principal" / "config.json").read_text())
pr_model = train_clip.build_model(train_clip.TrainConfig(**pr_dict))
pr_model.load_state_dict(load_checkpoint(tmp / "principal" / "best.pt")["model_state_dict"])
assert pr_dict["principal_axes"] is True and pr_model.principal_axes
older = train_clip.TrainConfig(**{k: v for k, v in pr_dict.items() if k != "principal_axes"})
assert older.principal_axes is False and not train_clip.build_model(older).principal_axes, "older runs: off"
keep = train_clip.build_datasets
train_clip.build_datasets = lambda cfg: (Synthetic(96, 1, True), Synthetic(40, 2, True))   # the same clouds each run
train_clip.main(["--set", *common, "epochs=1", f"out_dir=\"{tmp / 'pn_plain'}\""])
train_clip.main(["--set", *common, "epochs=1", "pointnet_checkpoint=true", f"out_dir=\"{tmp / 'pn_redo'}\""])
train_clip.build_datasets = keep
h_plain, h_redo = (json.loads((tmp / d / "history.json").read_text()) for d in ("pn_plain", "pn_redo"))
assert h_plain["train_loss"] == h_redo["train_loss"], "pointnet_checkpoint: the same training losses"
assert [v["r1/mean"] for v in h_plain["val"]] == [v["r1/mean"] for v in h_redo["val"]], "and the same val R@1"
rd_dict = json.loads((tmp / "pn_redo" / "config.json").read_text())
rd_model = train_clip.build_model(train_clip.TrainConfig(**rd_dict))
rd_model.load_state_dict(load_checkpoint(tmp / "pn_redo" / "best.pt")["model_state_dict"])
assert rd_dict["pointnet_checkpoint"] is True and all(e.checkpoint for e in rd_model.encoders.values())
no_option = {k: v for k, v in rd_dict.items() if k != "pointnet_checkpoint"}
assert not train_clip.TrainConfig(**no_option).pointnet_checkpoint, "configs written before the option: off"
train_clip.main(["--set", *common, "epochs=1", 'encoder="vn_pointnet"', "vn_widths=[8,16]", "vn_scalar_widths=[16,32]",
                 f"out_dir=\"{tmp / 'vn'}\""])
vh = json.loads((tmp / "vn" / "history.json").read_text())
assert len(vh["val"]) == 2 and all(np.isfinite(vh["train_loss"])), "a Vector Neurons run trains and validates"
vn_model = train_clip.build_model(train_clip.TrainConfig(**json.loads((tmp / "vn" / "config.json").read_text())))
vn_model.load_state_dict(load_checkpoint(tmp / "vn" / "best.pt")["model_state_dict"])
assert type(vn_model.encoders["dm"]).__name__ == "VNPointNetEncoder" and len(vn_model.encoders["dm"].blocks) == 2
assert train_clip.TrainConfig(**{k: v for k, v in pr_dict.items() if not k.startswith("vn_")}).vn_widths == \
    [32, 64, 128, 256], "older configs get the default widths"
ab_cfg = train_clip.TrainConfig(**json.loads((tmp / "ablation" / "config.json").read_text()))
rebuilt = train_clip.build_model(ab_cfg)
rebuilt.load_state_dict(load_checkpoint(tmp / "ablation" / "best.pt")["model_state_dict"])
assert ab_cfg.point_invariants and rebuilt.encoders["dm"][0][0].in_features == 3 + 1, "DM: positions + distance"
assert ab_cfg.drop_inputs == ["stars.age"] and rebuilt.encoders["stars"][0][0].in_features == 3 + 1 + 1, \
    "stars: positions + metallicity (age hidden) + distance"
assert train_clip.TrainConfig(**{k: v for k, v in json.loads((tmp / "ablation" / "config.json").read_text()).items()
                                 if k != "drop_inputs"}).drop_inputs == [], "older configs hide nothing"
for bad in ('inputs="no_positions"', 'loss_pairs="dm_anchor"', 'patch_centres="random"', "shared_encoder=true",
            'dm_aperture="r200c"', 'drop_inputs=["stars.x"]', 'drop_inputs=["stars.mass"]', 'drop_inputs=["age"]'):
    try:
        train_clip.TrainConfig.from_file(None, [bad])
        raise AssertionError(f"{bad} should be refused")
    except SystemExit:
        pass
try:
    train_clip.TrainConfig.from_file(None, ['modalities=["stars","dm"]', 'drop_inputs=["gas.sf"]'])
    raise AssertionError("hiding a column of a modality that is not trained should be refused")
except SystemExit:
    pass
print(f"PASS training run: loss {hist['train_loss'][0]:.2f} -> {hist['train_loss'][-1]:.2f}, val R@1 "
      f"{hist['val'][0]['r1/mean']:.0%} -> {hist['val'][-1]['r1/mean']:.0%} (chance {hist['val'][0]['chance/mean']:.0%})"
      "; resume continues the history; the shuffle control runs; a run without velocities, with a stars anchor, a "
      "shared encoder, EMA, point invariants and the stars' age hidden runs and its checkpoint reloads; PointNet++ "
      "and PointNeXt runs train "
      "and their checkpoints reload; a principal-axes run trains and reloads, and configs written before the option "
      "leave it off; a run with pointnet_checkpoint gives the plain run's losses and R@1 and reloads; a Vector "
      "Neurons run trains and reloads; unknown option values, bad PointNet++ levels, PointNeXt stages or Vector "
      "Neurons widths, principal axes with Vector Neurons and pointnet_checkpoint without PointNet are refused")

# ---------------------------------------------------------------- scoring finished runs, linear probes (Phase D)
import h5py                                                                            # noqa: E402
from galaxyclip3d.evaluation import embed, load_run, retrieval_scores, untrained_model  # noqa: E402
from galaxyclip3d.probes import PROPERTIES, auc, load_targets, lookup, r2, ridge_probe  # noqa: E402

cfg_run, model_run, ck_run = load_run(tmp / "run")
val_ds = Synthetic(40, 2, True)
ref = train_clip.evaluate(model_run, torch.utils.data.DataLoader(val_ds, 16), torch.device("cpu"), cfg_run, None)
got = embed({"run": model_run, "untrained": untrained_model(cfg_run)}, val_ds, batch_size=16, workers=0,
            per_cloud=lambda c: c[:, :3].std(0))
res = retrieval_scores(got["run"], got["mask"], got["mstar"], ks=tuple(cfg_run.ks), mass_bin_dex=cfg_run.mass_bin_dex)
assert set(ref) == set(res) and all(abs(res[k] - ref[k]) < 1e-6 for k in ref), "same numbers as the trainer's evaluate"
assert got["run"]["stars"].shape == (40, 32) and got["per_cloud"]["gas"].shape == (40, 3) and got["mask"].shape == (40, 4)
assert not np.allclose(got["run"]["stars"], got["untrained"]["stars"]), "the untrained model differs from the trained"
again = embed({"u": untrained_model(cfg_run)}, val_ds, batch_size=16, workers=0)
assert np.allclose(again["u"]["stars"], got["untrained"]["stars"]), "the untrained weights are seeded"
g = np.random.default_rng(5)
x = g.normal(size=(600, 20))
y = x @ g.normal(size=20) + 0.1 * g.normal(size=600)
fit = ridge_probe(x[:400], y[:400], x[400:500], y[400:500], x[500:], y[500:])
assert fit["val"] > 0.95 and fit["test"] > 0.95 and fit["n_test"] == 100, fit
noise = ridge_probe(x[:400], g.normal(size=400), x[400:500], g.normal(size=100), x[500:], g.normal(size=100))
assert noise["test"] < 0.1, noise
lab = (x[:, 0] > 0).astype(float)
cls = ridge_probe(x[:400], lab[:400], x[400:500], lab[400:500], x[500:], lab[500:], score="auc")
assert cls["test"] > 0.95, cls
assert auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0
assert auc(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1])) == 0.0
assert auc(np.array([0, 1, 0, 1]), np.array([.5, .5, .5, .5])) == 0.5
assert abs(r2(y, y) - 1.0) < 1e-12 and abs(r2(y, np.full_like(y, y.mean()))) < 1e-12
troot = tmp / "targets"
(troot / "snap_099").mkdir(parents=True)
with h5py.File(troot / "snap_099" / "targets.h5", "w") as f:
    grp = f.create_group("galaxies")
    grp["subhalo"] = np.array([3, 1, 2])
    for column in {c for c, _, _ in PROPERTIES.values()}:
        grp[column] = np.array([-15.0, -10.0, -15.0]) if column == "log_ssfr" else np.array([10.0, 0.0, -1.0])
tg = load_targets(troot, 99)
assert np.isclose(tg["vmax"][0], 1.0) and np.isnan(tg["vmax"][1:]).all(), "log10 of positive values, else NaN"
assert tg["log_mstar_total"][2] == -1.0, "properties stored as logs stay as they are"
assert tg["log_mgas_total"][0] == 10.0 and np.isnan(tg["log_mgas_total"][1:]).all(), "no gas (mass floor) is left out"
assert np.array_equal(tg["log_sfr"], [10.0, 0.0, -1.0]), "star-formation rates above the floor are kept"
assert np.isnan(tg["log_ssfr"][[0, 2]]).all() and tg["log_ssfr"][1] == -10.0 and \
    np.array_equal(tg["quenched"], [1.0, 0.0, 1.0]), "the sSFR floor becomes the yes/no property quenched"
lk = lookup(tg, [2, 3, 8])
assert lk["log_mstar_total"][0] == -1.0 and lk["log_mstar_total"][1] == 10.0 and np.isnan(lk["log_mstar_total"][2])
print("PASS Phase D helpers: a reloaded run scores exactly as the trainer's evaluation, one pass embeds several models "
      "and per-cloud statistics, the untrained model is seeded; ridge probes recover a linear signal (R^2 "
      f"{fit['test']:.3f}), not noise ({noise['test']:.3f}), and a class (AUC {cls['test']:.3f}); AUC, R^2, targets "
      "and lookup by subhalo behave")
# ---------------------------------------------------------------- Stage E: fewer modalities (no HI)
class Without(torch.utils.data.Dataset):
    """A dataset whose items lack one modality, as PointCloudDataset(modalities=...) returns them."""

    def __init__(self, ds, drop):
        self.ds, self.drop = ds, drop

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = {k: v for k, v in self.ds[i].items() if k != self.drop}
        item["mask"] = item["mask"].clone()
        item["mask"][MODALITIES.index(self.drop)] = False
        return item


train_clip.build_datasets = lambda cfg: (Without(Synthetic(96, 1, False), "hi"), Without(Synthetic(40, 2, True), "hi"))
train_clip.main(["--set", *common, "epochs=2", 'modalities=["gas","stars","dm"]', f"out_dir=\"{tmp / 'stage_e'}\""])
se = json.loads((tmp / "stage_e" / "history.json").read_text())
assert set(se["pair_loss"][0]) == {"stars-dm", "stars-gas", "dm-gas"}, "only the pairs of the kept modalities"
directions = {k.split("/", 1)[1] for k in se["val"][-1] if "->" in k}
assert directions == {f"{a}->{b}" for a in ("stars", "dm", "gas") for b in ("stars", "dm", "gas") if a != b}, \
    "only the 6 directions of stars, dark matter and gas are scored"
assert se["val"][-1]["r1/mean_no_gas_hi"] == se["val"][-1]["r1/mean"], "without HI both means are the same"
se_cfg, se_model, _ = load_run(tmp / "stage_e")
assert se_cfg.modalities == ["stars", "dm", "gas"], "kept in the order of MODALITIES"
assert set(se_model.encoders) == {"stars", "dm", "gas"}, "one encoder per kept modality; the checkpoint reloads"
val3 = Without(Synthetic(40, 2, True), "hi")
ref3 = train_clip.evaluate(se_model, torch.utils.data.DataLoader(val3, 16), torch.device("cpu"), se_cfg, None)
got3 = embed({"e": se_model}, val3, batch_size=16, workers=0)
res3 = retrieval_scores(got3["e"], got3["mask"], got3["mstar"], ks=tuple(se_cfg.ks), mass_bin_dex=se_cfg.mass_bin_dex)
assert set(got3["e"]) == {"stars", "dm", "gas"} and set(ref3) == set(res3)
assert all(abs(res3[k] - ref3[k]) < 1e-6 for k in ref3), "scoring helpers match the trainer without HI too"
old = embed({"run": model_run}, val3, batch_size=16, workers=0)                     # a four-modality run without HI
r_old = retrieval_scores(old["run"], old["mask"], old["mstar"])
r_all = retrieval_scores(got["run"], got["mask"], got["mstar"])
kept = [k for k in r_all if k.startswith("r1/") and "->" in k and "hi" not in k]
assert set(old["run"]) == {"stars", "dm", "gas"} and all(r_old[k] == r_all[k] for k in kept), \
    "a four-modality run scored without HI keeps its numbers for the other directions"
assert abs(r_old["r1/mean"] - np.mean([r_all[k] for k in kept])) < 1e-9, "its mean is over the 6 directions"
for bad in (['modalities=["stars"]'], ['modalities=["stars","xray"]'], ['modalities=["dm","dm"]'],
            ["min_particles=-1"], ["points=4096", "min_particles=2048"],
            ['modalities=["dm","gas"]', 'loss_pairs="stars_anchor"']):
    try:
        train_clip.TrainConfig.from_file(None, bad)
        raise AssertionError(f"{bad} should be refused")
    except SystemExit:
        pass
print("PASS Stage E: a run without HI trains and scores only stars, dark matter and gas (3 pair losses, 6 "
      "directions), reloads with three encoders and scores the same with the Phase D helpers; a four-modality run "
      "scored without HI keeps its numbers for the other directions; too few, unknown or repeated modalities, a "
      "threshold below the points and a stars anchor without stars are refused")
shutil.rmtree(tmp, ignore_errors=True)
print("ALL TESTS PASSED")
