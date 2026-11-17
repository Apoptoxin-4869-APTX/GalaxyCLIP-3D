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
from galaxyclip3d.models.clip import invariant_features                                # noqa: E402
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
print("PASS orientation: lengths and r.v kept, scalars untouched, one matrix per modality (or one shared), "
      "reproducible with a seed, reflections with mirror, empty clouds stay empty")

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
                    ("pointnet", {"point_invariants": True}), ("patch", {"point_invariants": True})):
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
lt = MultiModalCLIP(DIMS, "pointnet", embed_dim=8, encoder_dim=8, learn_temperature=True, pointnet_widths=(8,))
assert isinstance(lt.logit_scale, torch.nn.Parameter) and abs(float(lt.temperature().detach()) - 0.07) < 1e-6
print("PASS encoders: shapes, unit-length embeddings, same output for shuffled points, masked clouds finite, "
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
print(f"PASS ablation options: dropped input columns are ignored, one shared PointNet trunk with its own first layer "
      f"per modality (fewer parameters, checkpoint round trip), radial patch centres put {inner['radial']:.0%} of "
      f"centres in the inner half of a concentrated cloud against {inner['fps']:.0%} for plain farthest-point sampling; "
      f"point invariants are unchanged by rotation and mirroring, exact on a known point, zero for masked clouds, and "
      f"carry no velocities when velocities are dropped")


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
                 "shared_encoder=true", "ema_decay=0.9", "point_invariants=true", f"out_dir=\"{tmp / 'ablation'}\""])
ab = json.loads((tmp / "ablation" / "history.json").read_text())
assert set(ab["pair_loss"][0]) == {"stars-dm", "stars-gas", "stars-hi"}, "the anchor trains only the stars pairs"
assert "r1/dm->gas" in ab["val"][-1] and len(ab["val"]) == 2, "validation keeps every direction"
ab_cfg = train_clip.TrainConfig(**json.loads((tmp / "ablation" / "config.json").read_text()))
rebuilt = train_clip.build_model(ab_cfg)
rebuilt.load_state_dict(load_checkpoint(tmp / "ablation" / "best.pt")["model_state_dict"])
assert ab_cfg.point_invariants and rebuilt.encoders["dm"][0][0].in_features == 3 + 1, "DM: positions + distance"
for bad in ('inputs="no_positions"', 'loss_pairs="dm_anchor"', 'patch_centres="random"', "shared_encoder=true"):
    try:
        train_clip.TrainConfig.from_file(None, [bad])
        raise AssertionError(f"{bad} should be refused")
    except SystemExit:
        pass
print(f"PASS training run: loss {hist['train_loss'][0]:.2f} -> {hist['train_loss'][-1]:.2f}, val R@1 "
      f"{hist['val'][0]['r1/mean']:.0%} -> {hist['val'][-1]['r1/mean']:.0%} (chance {hist['val'][0]['chance/mean']:.0%})"
      "; resume continues the history; the shuffle control runs; a run without velocities, with a stars anchor, a "
      "shared encoder, EMA and point invariants runs and its checkpoint reloads; unknown option values are refused")

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
shutil.rmtree(tmp, ignore_errors=True)
print("ALL TESTS PASSED")
