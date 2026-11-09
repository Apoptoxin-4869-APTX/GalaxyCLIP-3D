"""Tests of the point-cloud build on two small synthetic snapshots (99 at z = 0, 91 at a = 0.9) with TNG's file layout:
a dataset built from snapshot 99, then snapshot 91 added to it, with merger-tree branches given offline.
Run: PYTHONPATH=pylibs python3 -s tests/test_pointclouds.py   (numpy, h5py; the PDF check also needs matplotlib, the
loader part torch)
"""
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from galaxyclip3d.pointclouds import lineage                                          # noqa: E402
from galaxyclip3d.pointclouds import physics as ph                                     # noqa: E402
from galaxyclip3d.pointclouds.config import Config                                    # noqa: E402
from galaxyclip3d.pointclouds.extract import extract, inclusion_probabilities         # noqa: E402
from galaxyclip3d.pointclouds.sampling import FEATURES, MODALITIES, draw, random_rotation  # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, shard_path, snapshot_dir      # noqa: E402
from galaxyclip3d.pointclouds.select import assign_splits, select, shard_plan         # noqa: E402
from galaxyclip3d.pointclouds.snapshot import Snapshot, read_catalog                  # noqa: E402
from galaxyclip3d.pointclouds.store import Shard, write_shard                         # noqa: E402

H, BOX = 0.6774, 1000.0
TMP = Path(tempfile.mkdtemp(prefix="pc_test_"))
RNG = np.random.default_rng(1)
# subhalo: group, central position (ckpc/h), (gas, dm, stars) bound counts, R1/2,* (ckpc/h), SubhaloFlag
SUBHALOS = [
    (0, (995.0, 500.0, 500.0), (7000, 9000, 6000), 5.0, 1),    # central at the box edge, capped (cap 3000)
    (0, (930.0, 500.0, 500.0), (0, 5000, 5000), 3.0, 1),       # its satellite, without gas
    (1, (300.0, 300.0, 300.0), (2500, 4500, 4400), 4.0, 1),    # central whose gas stays below the cap
    (2, (600.0, 100.0, 800.0), (100, 5000, 1000), 4.0, 1),     # too few stars
    (3, (100.0, 900.0, 100.0), (5000, 5000, 5000), 4.0, 0),    # SubhaloFlag = 0
    (4, (500.0, 500.0, 100.0), (5000, 5000, 5000), 0.2, 1),    # too small (0.3 kpc)
]
SUBHALOS_91 = [list(s) for s in SUBHALOS]
SUBHALOS_91[3][2] = (100, 5000, 4500)                          # at snapshot 91 subhalo 3 has enough stars (no gas)
SUBHALOS_91 = [tuple(s) for s in SUBHALOS_91]
A_91 = 0.9                                                     # scale factor of the synthetic snapshot 91
N_WIND = 200                                                   # wind cells among subhalo 0's star particles
GROUP_FIRST = [0, 2, 3, 4, 5]
# merger-tree main branches (snapshot, subhalo) as the TNG API's sublink/simple.json would give them
BRANCHES_99 = {0: [[91, 0], [99, 0]], 1: [[91, 1], [99, 1]], 2: [[99, 2]]}
BRANCHES_91 = {0: [[90, 3], [91, 0], [95, 0], [99, 0]],       # ends in subhalo 0 of halo 0 at z = 0
               1: [[91, 1], [99, 1]],                         # a satellite ending in halo 0 too
               2: [[91, 2], [99, 5]],                         # ends in subhalo 5, halo 4: no selected galaxy at z = 0
               3: [[90, 7], [91, 3]]}                         # branch ends before z = 0: its own halo (91, 2) decides


def blob(n, centre, sigma):
    return (np.asarray(centre) + RNG.normal(scale=sigma, size=(n, 3))) % BOX


def make_snapshot(root: Path, snap: int = 99, time_a: float = 1.0, subhalos=SUBHALOS) -> dict:
    """Two snapshot files (the split falls inside subhalo 0's blocks), one catalog file, the offsets file."""
    parts = {0: [], 1: [], 4: []}
    for s, (grp, c, (ng, nd, ns), rh, flag) in enumerate(subhalos):
        g = {"Coordinates": blob(ng, c, 15.0), "Velocities": RNG.normal(100, 50, (ng, 3)).astype(np.float32),
             "Masses": RNG.uniform(0.6, 1.6, ng).astype(np.float32) * 5.76e-6,
             "Density": RNG.lognormal(-14, 2, ng).astype(np.float32),
             "InternalEnergy": RNG.lognormal(np.log(300), 1.5, ng).astype(np.float32),     # about 1e4 K
             "ElectronAbundance": RNG.uniform(0, 1.2, ng).astype(np.float32),
             "NeutralHydrogenAbundance": (RNG.uniform(0, 1, ng) ** 4).astype(np.float32),
             "StarFormationRate": np.where(RNG.random(ng) < 0.2, 0.01, 0.0).astype(np.float32),
             "GFM_Metallicity": RNG.uniform(1e-5, 0.03, ng).astype(np.float32),
             "MagneticField": RNG.normal(0, 1e-4, (ng, 3)).astype(np.float32)}
        d = {"Coordinates": blob(nd, c, 12.0), "Velocities": RNG.normal(0, 80, (nd, 3)).astype(np.float32)}
        sft = RNG.uniform(0.1, 1.0, ns).astype(np.float32)
        if s == 0:
            sft[:N_WIND] = -RNG.uniform(0.1, 1.0, N_WIND)
        st = {"Coordinates": blob(ns, c, 8.0), "Velocities": RNG.normal(0, 60, (ns, 3)).astype(np.float32),
              "Masses": RNG.uniform(3.0, 5.76, ns).astype(np.float32) * 1e-6,
              "GFM_InitialMass": np.full(ns, 5.76e-6, np.float32), "GFM_Metallicity": RNG.uniform(0, 0.04, ns).astype(np.float32),
              "GFM_StellarFormationTime": sft}
        for t, dd in ((0, g), (1, d), (4, st)):
            parts[t].append(dd)
    full = {t: {k: np.concatenate([p[k] for p in parts[t]]) for k in parts[t][0]} for t in parts}
    total = {t: len(full[t]["Coordinates"]) for t in full}
    split = {0: 3000, 1: 5000, 4: 2500}                       # rows of each type in file 0
    sdir = root / "snapshots" / f"{snap:03d}"
    sdir.mkdir(parents=True)
    for i in range(2):
        with h5py.File(sdir / f"snap_{snap:03d}.{i}.hdf5", "w") as f:
            npart = np.zeros(6, np.int64)
            for t in full:
                lo, hi = (0, split[t]) if i == 0 else (split[t], total[t])
                npart[t] = hi - lo
                g = f.create_group(f"PartType{t}")
                for k, v in full[t].items():
                    g.create_dataset(k, data=v[lo:hi])
            tot = np.zeros(6, np.int64)
            for t in full:
                tot[t] = total[t]
            f.create_group("Header").attrs.update(dict(
                HubbleParam=H, Time=time_a, Redshift=1 / time_a - 1, BoxSize=BOX, Omega0=0.3089, OmegaLambda=0.6911,
                NumPart_ThisFile=npart,
                NumPart_Total=tot, NumPart_Total_HighWord=np.zeros(6, np.int64), NumFilesPerSnapshot=2,
                MassTable=np.array([0, 3.07e-5, 0, 0, 0, 0])))
    lens = np.array([[ng, nd, 0, 0, ns, 0] for _, _, (ng, nd, ns), _, _ in subhalos], dtype=np.int64)
    gdir = root / "groupcats" / f"{snap:03d}"
    gdir.mkdir(parents=True)
    with h5py.File(gdir / f"fof_subhalo_tab_{snap:03d}.0.hdf5", "w") as f:
        f.create_group("Header").attrs.update(dict(Ngroups_ThisFile=len(GROUP_FIRST), Nsubgroups_ThisFile=len(subhalos),
                                                   Redshift=1 / time_a - 1, Time=time_a))
        f.create_group("Group").create_dataset("GroupFirstSub", data=np.array(GROUP_FIRST, np.int32))
        sg = f.create_group("Subhalo")
        rh = np.zeros((len(subhalos), 6), np.float32)
        rh[:, 4] = [s[3] for s in subhalos]
        mir = np.zeros((len(subhalos), 6), np.float32)
        mir[:, 4] = [0.01, 0.003, 0.02, 0.001, 0.01, 0.01]
        sg.create_dataset("SubhaloLenType", data=lens.astype(np.int32))
        sg.create_dataset("SubhaloFlag", data=np.array([s[4] for s in subhalos], np.uint8))
        sg.create_dataset("SubhaloHalfmassRadType", data=rh)
        sg.create_dataset("SubhaloMassInRadType", data=mir)
        sg.create_dataset("SubhaloSFRinRad", data=np.array([0.5, 0.0, 1.0, 0.0, 0.1, 0.1], np.float32))
        sg.create_dataset("SubhaloGrNr", data=np.array([s[0] for s in subhalos], np.int32))
        sg.create_dataset("SubhaloPos", data=np.array([s[1] for s in subhalos], np.float32))
        sg.create_dataset("SubhaloVel", data=np.tile(np.array([100.0, 100.0, 100.0], np.float32), (len(subhalos), 1)))
        sg.create_dataset("SubhaloVmax", data=np.array([150.0, 80.0, 120.0, 50.0, 60.0, 60.0], np.float32))
    starts = np.vstack([np.zeros(6, np.int64), np.cumsum(lens, axis=0)[:-1]])
    odir = root / "offsets"
    odir.mkdir(exist_ok=True)
    with h5py.File(odir / f"offsets_{snap:03d}.hdf5", "w") as f:
        fo = np.zeros((2, 6), np.int64)
        for t in split:
            fo[1, t] = split[t]
        f.create_dataset("FileOffsets/SnapByType", data=fo)
        f.create_dataset("Subhalo/SnapByType", data=starts)
    return {"full": full, "starts": starts, "lens": lens}


def region_truth(truth, s, t, radius_kpc, scale=1.0):
    """Rows of subhalo s, type t, inside the sphere (wind removed), straight from the synthetic arrays."""
    a, n = truth["starts"][s, t], truth["lens"][s, t]
    d = {k: v[a:a + n] for k, v in truth["full"][t].items()}
    r = np.linalg.norm(ph.relative_positions(d["Coordinates"], SUBHALOS[s][1], BOX) * scale / H, axis=1)
    keep = r <= radius_kpc
    if t == 4:
        keep &= d["GFM_StellarFormationTime"] > 0
    return {k: v[keep] for k, v in d.items()}


def run(*cmd):
    p = subprocess.run([sys.executable, "-s", *map(str, cmd)], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
    assert p.returncode == 0, (cmd, p.stdout[-2000:], p.stderr[-2000:])
    return p.stdout


truth = make_snapshot(TMP / "snap")
truth91 = make_snapshot(TMP / "snap", 91, A_91, SUBHALOS_91)
cfg = Config(root=str(TMP / "snap"), out=str(TMP / "out"), budget=256, cap=3000, shard_size=2, hi_min_eff=50.0,
             chunk_rows=1500)
snap = Snapshot(cfg.root, cfg.snap)

# ---------------------------------------------------------------- selection
table = select(cfg, snap)
assert table["subhalo"].tolist() == [0, 1, 2], table["subhalo"]
assert table["central"].tolist() == [True, False, True]
assert (table["snapshot"] == 99).all() and np.allclose(table["redshift"], 0.0)
assert np.allclose(table["rhalf_star_kpc"], np.array([5.0, 3.0, 4.0]) / H)
assert shard_plan(table, 2).max() == 1
code = assign_splits(np.arange(1000) // 3, RNG.normal(9, 1, 1000), (0.8, 0.1, 0.1), 4869)
share = np.bincount(code, minlength=3) / 1000
assert abs(share[0] - 0.8) < 0.03 and abs(share[1] - 0.1) < 0.03, share
t91 = select(dataclasses.replace(cfg, snap=91), Snapshot(cfg.root, 91))
assert t91["subhalo"].tolist() == [0, 1, 2, 3] and (t91["snapshot"] == 91).all()
assert np.allclose(t91["redshift"], 1 / A_91 - 1)
assert np.allclose(t91["rhalf_star_kpc"], np.array([5.0, 3.0, 4.0, 4.0]) * A_91 / H), "comoving radii times a/h"
try:
    read_catalog(cfg.root, 91, [], ["SubhaloGrNr"], expect_redshift=0.0)
    raise AssertionError("a catalog of another redshift must be refused")
except ValueError:
    pass
print("PASS selection: flag, star/DM counts and size filters, snapshot and redshift recorded, radii in physical kpc "
      "at a < 1; a catalog of another snapshot is refused; splits close to 80/10/10")

# ---------------------------------------------------------------- lineage and splits across snapshots
group = RNG.integers(0, 400, 1500)
lm = RNG.normal(9.5, 0.8, 1500)
units = lineage.unit_keys(np.full(1500, 99), group)
first, reg, n_new = lineage.assign(99, units, lm, {}, (0.8, 0.1, 0.1), 4869)
assert np.array_equal(first, assign_splits(group, lm, (0.8, 0.1, 0.1), 4869)), "first snapshot = the old halo split"
assert n_new == len(np.unique(group)) and len(reg) == n_new
again, reg2, n_again = lineage.assign(99, units, lm, reg, (0.8, 0.1, 0.1), 4869)
assert np.array_equal(again, first) and n_again == 0 and reg2 == reg
b = lineage.parse_simple({"Main": [[99, 5], [91, 2], [95, 4]], "Mergers": [[80, 1]]})
assert b[:, 0].tolist() == [91, 95, 99]
br = {int(k): lineage.parse_simple({"Main": v}) for k, v in BRANCHES_91.items()}
desc, ok = lineage.z0_descendants(91, np.array([0, 1, 2, 3, 9]), br)
assert desc.tolist() == [0, 1, 5, -1, -1] and ok.tolist() == [True, True, True, True, False]
z0_grnr = np.array([s[0] for s in SUBHALOS], np.int32)
usnap, ugroup, zg = lineage.split_units(91, np.array([0, 0, 1, 2, 3]), desc, z0_grnr)
assert usnap.tolist() == [99, 99, 99, 91, 91] and ugroup.tolist() == [0, 0, 4, 2, 3] and zg.tolist() == [0, 0, 4, -1, -1]


class FakeClient:
    api_base = "https://example.invalid/api/TNG50-1"

    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), 0

    def get_json(self, path):
        self.calls += 1
        s = int(path.split("/")[3])
        if s in self.fail:
            raise RuntimeError("HTTP 504")
        return {"Main": BRANCHES_91[s], "Mergers": []}


cache = TMP / "lineage_test.h5"
fc = FakeClient(fail={3})
got = lineage.fetch_branches(91, [0, 1, 2, 3], cache, fc, workers=2, log=lambda *a: None)
assert sorted(got) == [0, 1, 2] and fc.calls == 4
fc2 = FakeClient()
got = lineage.fetch_branches(91, [0, 1, 2, 3], cache, fc2, workers=2, log=lambda *a: None)
assert sorted(got) == [0, 1, 2, 3] and fc2.calls == 1, "cached branches are not fetched again"
assert all(np.array_equal(lineage.read_branches(cache)[s], br[s]) for s in br)
from galaxyclip3d.pointclouds.tng_api import ApiError                                  # noqa: E402


class StatusClient(FakeClient):
    def __init__(self, status, fatal=False):
        super().__init__()
        self.status, self.fatal = status, fatal

    def get_json(self, path):
        raise ApiError("test", self.status, fatal=self.fatal)


got = lineage.fetch_branches(91, [7], TMP / "lineage_404.h5", StatusClient(404), log=lambda *a: None)
assert len(got[7]) == 0, "no tree at the API: an empty branch, not a failure to retry"
assert lineage.z0_descendants(91, np.array([7]), got)[1].tolist() == [False]
try:
    lineage.fetch_branches(91, [8, 9], TMP / "lineage_401.h5", StatusClient(401, fatal=True), log=lambda *a: None)
    raise AssertionError("a rejected key must stop the fetching")
except ApiError:
    pass
print("PASS lineage: the first snapshot's split equals the old halo split, rerunning changes nothing, branches parsed "
      "and cached (failed requests retried on the next run), descendants turned into z = 0 halos, own halo without one")

# ---------------------------------------------------------------- extraction
rows = {s: {k: v[i] for k, v in table.items()} for i, s in enumerate(table["subhalo"])}
e0 = extract(snap, cfg, rows[0])
r_ext = cfg.extract_radius * rows[0]["rhalf_star_kpc"]
true_stars = region_truth(truth, 0, 4, r_ext)
assert e0["counts"]["n_stars_r10"] == len(true_stars["Masses"])
assert e0["counts"]["n_stars_r10"] <= 6000 - N_WIND
assert abs(e0["counts"]["n_stars_stored"] - 3000) < 200, e0["counts"]["n_stars_stored"]
assert np.abs(e0["stars"]["pos"].mean(0)).max() < 2.0, "positions must be wrapped around the centre"
assert np.linalg.norm(e0["stars"]["pos"], axis=1).max() <= r_ext * 1.0001
est = (e0["stars"]["mass"] * e0["stars"]["wcorr"]).sum()
true_m = true_stars["Masses"].astype(np.float64).sum() * 1e10 / H
assert abs(est / true_m - 1) < 0.03, (est, true_m)
true_gas = region_truth(truth, 0, 0, r_ext)
true_hi = (ph.X_H * true_gas["NeutralHydrogenAbundance"] * true_gas["Masses"]).astype(np.float64).sum() * 1e10 / H
est_hi = (e0["gas"]["m_hi"] * e0["gas"]["wcorr"]).sum()
assert abs(est_hi / true_hi - 1) < 0.05, (est_hi, true_hi)
assert np.allclose(e0["dm"]["wcorr"], e0["dm"]["wcorr"][0]), "uniform capping for dark matter"
counts = [e0["counts"][f"n_dm_r{k:g}"] for k in cfg.radii]
assert counts == sorted(counts)
pi = inclusion_probabilities(RNG.pareto(1.0, 10000) + 0.01, 500)
assert abs(pi.sum() - 500) < 1e-6 and pi.max() <= 1
print("PASS extraction: box edge wrapped, sphere cut, wind removed, capping keeps the mass and HI totals unbiased")

e1 = extract(snap, cfg, rows[1])
assert len(e1["gas"]["pos"]) == 0 and e1["counts"]["hi_eff_r10"] == 0.0 and e1["counts"]["n_gas_r5"] == 0
e2 = extract(snap, cfg, rows[2])
g2 = region_truth(truth, 2, 0, cfg.extract_radius * rows[2]["rhalf_star_kpc"])
assert e2["counts"]["n_gas_stored"] == len(g2["Masses"]) and np.all(e2["gas"]["wcorr"] == 1)
mu = 4 / (1 + 3 * 0.76 + 4 * 0.76 * g2["ElectronAbundance"].astype(np.float64))
t_k = (5 / 3 - 1) * g2["InternalEnergy"].astype(np.float64) * 1e10 * mu * 1.67262192e-24 / 1.380649e-16
assert np.allclose(np.sort(e2["gas"]["log_t"]), np.sort(np.log10(t_k)).astype(np.float32), rtol=1e-5)
rel = ph.relative_positions(g2["Coordinates"], SUBHALOS[2][1], BOX) / H
assert np.allclose(np.sort(e2["gas"]["pos"][:, 0]), np.sort(rel[:, 0]).astype(np.float32), atol=1e-4)
assert np.allclose(e2["gas"]["vel"].mean(0), (g2["Velocities"].astype(np.float64) - 100).mean(0), atol=1e-3)
print("PASS units and derived fields: positions in kpc, velocities relative to the galaxy, temperatures; "
      "a galaxy without gas gets empty gas arrays")

snap91 = Snapshot(cfg.root, 91)
cfg91 = dataclasses.replace(cfg, snap=91, cap=10 ** 9)          # uncapped, to compare particle by particle
row91 = {k: v[2] for k, v in t91.items()}
e91 = extract(snap91, cfg91, row91)
r91 = cfg.extract_radius * row91["rhalf_star_kpc"]
g91 = region_truth(truth91, 2, 0, r91, scale=A_91)
assert e91["counts"]["n_gas_stored"] == len(g91["Masses"])
rel91 = ph.relative_positions(g91["Coordinates"], SUBHALOS[2][1], BOX) * A_91 / H
assert np.allclose(np.sort(e91["gas"]["pos"][:, 0]), np.sort(rel91[:, 0]).astype(np.float32), atol=1e-4)
assert np.allclose(e91["gas"]["vel"].mean(0), (g91["Velocities"].astype(np.float64) * np.sqrt(A_91) - 100).mean(0),
                   atol=1e-3)
s91 = region_truth(truth91, 2, 4, r91, scale=A_91)
u91 = ph.Units(H, A_91, BOX, 0.3089, 0.6911)
want_age = ph.cosmic_time_gyr(A_91, u91) - ph.cosmic_time_gyr(s91["GFM_StellarFormationTime"].astype(np.float64), u91)
assert np.allclose(np.sort(e91["stars"]["age"]), np.sort(np.maximum(want_age, 0)).astype(np.float32), atol=1e-4)
assert e91["stars"]["age"].max() < ph.cosmic_time_gyr(A_91, u91), "ages count from the snapshot's own time"
print("PASS a = 0.9: positions times a/h, velocities times sqrt(a), stellar ages from the snapshot's time")

# ---------------------------------------------------------------- shards, sampling
path = TMP / "shard_test.h5"
gal = {k: v[[0, 1, 2]] for k, v in table.items()}
write_shard(path, gal, [e0, e1, e2], {"config": cfg.to_json()})
sh = Shard(path)
for i, e in enumerate([e0, e1, e2]):
    for typ in ("gas", "dm", "stars"):
        back = sh.particles(i, typ)
        for f, v in e[typ].items():
            assert np.array_equal(back[f], v), (i, typ, f)
print("PASS shard files: everything written comes back unchanged")

parts0 = {t: sh.particles(0, t) for t in ("stars", "dm", "gas")}
radius = cfg.train_radius * rows[0]["rhalf_star_kpc"]
c = draw(parts0, radius, 150.0, cfg.budget, np.random.default_rng(3))
for m in MODALITIES:
    assert c[m].shape == (cfg.budget, len(FEATURES[m])) and c[m].dtype == np.float32
gas_xyz = {tuple(x) for x in c["gas"][:, :3]}
assert not any(tuple(x) in gas_xyz for x in c["hi"][:, :3]), "gas and HI share coordinates"
assert np.abs(c["stars"][:, :3]).max() <= 1.0 + 1e-6
rot = random_rotation(np.random.default_rng(5))
assert np.allclose(rot @ rot.T, np.eye(3)) and abs(np.linalg.det(rot) - 1) < 1e-9
r1 = np.random.default_rng(5)
c_rot = draw(parts0, radius, 150.0, cfg.budget, r1, rotate=True)
r2 = np.random.default_rng(5)
r2.normal(size=4)                                            # the draws the rotation consumed
c_plain = draw(parts0, radius, 150.0, cfg.budget, r2, rotate=False)
for m in MODALITIES:
    assert np.allclose(c_rot[m][:, :3], c_plain[m][:, :3] @ rot.T, atol=1e-5), m
    assert np.allclose(c_rot[m][:, 3:6], c_plain[m][:, 3:6] @ rot.T, atol=1e-5), m
    assert np.allclose(c_rot[m][:, 6:], c_plain[m][:, 6:]), m
c1 = draw({t: sh.particles(1, t) for t in ("stars", "dm", "gas")}, cfg.train_radius * rows[1]["rhalf_star_kpc"], 80.0,
          cfg.budget, np.random.default_rng(0))
assert not c1["gas"].any() and not c1["hi"].any() and c1["stars"].any()
print("PASS sampling: shapes, positions scaled to the sphere, no gas/HI coordinate sharing, one rotation for all "
      "modalities, empty clouds for missing gas")

# ---------------------------------------------------------------- command-line pipeline: snapshot 99, then 91 added
cfg_path = TMP / "cfg.json"
cfg_path.write_text(cfg.to_json())
(TMP / "branches_99.json").write_text(json.dumps(BRANCHES_99))
(TMP / "branches_91.json").write_text(json.dumps(BRANCHES_91))
out = TMP / "out"
s99 = snapshot_dir(out, 99)
out.mkdir()       # units recorded beforehand keep their split (a few units alone would all be dealt to train): z = 0
PRESET = {int(lineage.unit_keys(99, 0)): (0, 99), int(lineage.unit_keys(99, 1)): (1, 99),    # halos 0 -> train,
          int(lineage.unit_keys(99, 4)): (2, 99)}                                           # 1 -> val, 4 -> test
lineage.save_splits(out, PRESET, None, ("train", "val", "test"))
run(REPO / "tools" / "build_selection.py", "--config", cfg_path, "--branches", TMP / "branches_99.json")
sel99 = read_table(s99 / "selection.h5")
assert sel99["split"].tolist() == [0, 0, 1], "a halo's galaxies share a split; recorded units keep theirs"
assert (sel99["unit_snap"] == 99).all() and np.array_equal(sel99["unit_group"], sel99["group"])
assert np.array_equal(sel99["z0_subhalo"], sel99["subhalo"]) and sel99["has_branch"].all()
reg99, z0 = lineage.load_splits(out)
assert reg99 == PRESET and np.array_equal(z0, z0_grnr), "snapshot 99's SubhaloGrNr kept in splits.h5"
assert Config.load(out / "config.json").snap is None and Config.load(s99 / "config.json").snap == 99
(s99 / "shards").mkdir(exist_ok=True)
(s99 / "shards" / "shard_0000.h5.part").write_text("leftover of a killed job")
log = run(REPO / "tools" / "build_pointclouds.py", "--out", out, "--snap", 99, "--all", "--workers", "2")
assert shard_path(out, 99, 0).exists() and shard_path(out, 99, 1).exists(), log
assert not list((s99 / "shards").glob("*.part"))
assert "0 shards to build" in run(REPO / "tools" / "build_pointclouds.py", "--out", out, "--snap", 99, "--all")
sh0 = Shard(shard_path(out, 99, 0))
assert sh0.file.attrs["snapshot"] == 99 and (sh0.galaxies()["snapshot"] == 99).all()
sh0.close()
run(REPO / "tools" / "finalize_pointclouds.py", "--out", out, "--snap", 99, "--stats-galaxies", "5")
assert read_table(out / "index.h5")["subhalo"].tolist() == [0, 1, 2]
stats = json.loads((out / "feature_stats.json").read_text())
assert set(stats) >= set(MODALITIES)
try:
    import matplotlib                                                             # noqa: F401
    want_pdf = ["--pdf", str(TMP / "clouds.pdf"), "--galaxies", "3"]
except ImportError:
    want_pdf = []
check = subprocess.run([sys.executable, "-s", str(REPO / "tools" / "check_pointclouds.py"), "--out", str(out),
                        "--snap", "99", *want_pdf], capture_output=True, text=True, env={**os.environ})
assert check.returncode == 0 and "no problems" in check.stdout, (check.stdout, check.stderr)
assert not want_pdf or (TMP / "clouds.pdf").stat().st_size > 10000
out2 = TMP / "out2"
run(REPO / "tools" / "build_selection.py", "--config", cfg_path, "--out", out2, "--branches", TMP / "branches_99.json")
run(REPO / "tools" / "build_pointclouds.py", "--out", out2, "--snap", 99, "--all")
for s in range(2):
    a, b = Shard(shard_path(out, 99, s)), Shard(shard_path(out2, 99, s))
    for typ in ("gas", "dm", "stars"):
        for f in a.file[typ]:
            assert np.array_equal(a.file[typ][f][()], b.file[typ][f][()]), (s, typ, f)
bad = dataclasses.replace(cfg, budget=512)
(TMP / "cfg_bad.json").write_text(bad.to_json())
p = subprocess.run([sys.executable, "-s", str(REPO / "tools" / "build_selection.py"), "--config",
                    str(TMP / "cfg_bad.json"), "--snap", "91", "--branches", str(TMP / "branches_91.json")],
                   capture_output=True, text=True)
assert p.returncode != 0 and "different settings" in p.stderr
print("PASS pipeline, snapshot 99: selection with splits by halo, shards with a leftover .part file, rerun skips "
      "finished shards, shards record their snapshot, index, feature statistics, frozen sets, quality checks and PDF; "
      "two builds are identical; a snapshot with other settings is refused")

stats_before = (out / "feature_stats.json").read_text()
frozen_before = {f.name: f.read_bytes() for f in (s99 / "frozen").glob("*.h5")}
run(REPO / "tools" / "build_selection.py", "--config", cfg_path, "--snap", 91, "--branches", TMP / "branches_91.json")
s91dir = snapshot_dir(out, 91)
sel91 = read_table(s91dir / "selection.h5")
reg, _ = lineage.load_splits(out)
assert sel91["subhalo"].tolist() == [0, 1, 2, 3] and (sel91["snapshot"] == 91).all()
assert sel91["z0_subhalo"].tolist() == [0, 1, 5, -1] and sel91["z0_group"].tolist() == [0, 0, 4, -1]
assert sel91["unit_snap"].tolist() == [99, 99, 99, 91] and sel91["unit_group"].tolist() == [0, 0, 4, 2]
assert sel91["split"][0] == sel91["split"][1] == sel99["split"][0], "versions of one galaxy share a split"
assert sel91["split"].tolist()[:3] == [0, 0, 2], "the units' recorded splits, z = 0 halos 0, 0 and 4"
assert len(reg) == 4 and reg[int(lineage.unit_keys(91, 2))][1] == 91, "the unit without a z = 0 descendant is new"
assert {k: v for k, v in reg.items() if v[1] == 99} == reg99, "earlier units keep their split"
run(REPO / "tools" / "build_selection.py", "--config", cfg_path, "--snap", 91, "--branches", TMP / "branches_91.json")
assert lineage.load_splits(out)[0] == reg, "running the selection again changes nothing"
run(REPO / "tools" / "build_pointclouds.py", "--out", out, "--snap", 91, "--all")
run(REPO / "tools" / "finalize_pointclouds.py", "--out", out, "--snap", 91, "--stats-galaxies", "5")
for sn in (99, 91):                       # frozen sets at the stored sphere as well, where every test galaxy is usable
    run(REPO / "tools" / "finalize_pointclouds.py", "--out", out, "--snap", sn, "--radius", 10)
every = read_table(out / "index.h5")
assert every["snapshot"].tolist() == [91, 91, 91, 91, 99, 99, 99] and every["subhalo"].tolist() == [0, 1, 2, 3, 0, 1, 2]
assert (out / "feature_stats.json").read_text() == stats_before, "feature statistics stay fixed"
assert {f.name: (s99 / "frozen" / f.name).read_bytes() for f in (s99 / "frozen").glob("*_r5.h5")} == frozen_before, \
    "snapshot 99 untouched"
summary = json.loads((out / "snapshots.json").read_text())
assert sorted(summary) == ["91", "99"] and abs(summary["91"]["redshift"] - (1 / A_91 - 1)) < 1e-9
check = subprocess.run([sys.executable, "-s", str(REPO / "tools" / "check_pointclouds.py"), "--out", str(out),
                        "--snap", "91"], capture_output=True, text=True, env={**os.environ})
assert check.returncode == 0 and "no problems" in check.stdout, (check.stdout, check.stderr)
print("PASS adding snapshot 91: units from merger-tree branches (z = 0 halo, or its own halo without one), a galaxy's "
      "versions share a split, new units recorded, rerun changes nothing, dataset index over both snapshots, "
      "feature statistics and snapshot 99's files unchanged, quality checks")

try:
    import torch                                                                  # noqa: F401
except ImportError:
    print("SKIP loader: no torch in this environment (run on a GPU node's venv)")
else:
    from galaxyclip3d.dataset import MassBalancedSampler, PointCloudDataset
    R = 10.0                                              # every synthetic galaxy is usable at the stored sphere
    ds = PointCloudDataset(out, "train", radius=R)
    assert len(ds) > 0
    item = ds[0]
    assert all(item[m].shape == (cfg.budget, len(FEATURES[m])) for m in MODALITIES) and item["mask"].shape == (4,)
    assert len(list(iter(MassBalancedSampler(ds)))) == len(ds)
    for snaps in ([91], [99]):
        d1 = PointCloudDataset(out, "train", radius=R, snapshots=snaps)
        assert all(d1[i]["snapshot"] == snaps[0] for i in range(len(d1)))
    sizes = {sp: len(PointCloudDataset(out, sp, radius=R)) for sp in ("train", "val", "test")}
    assert sum(sizes.values()) == len(every["subhalo"]) and min(sizes.values()) > 0, sizes
    seen = set()
    for split in ("val", "test"):
        dv = PointCloudDataset(out, split, radius=R)
        for i in range(len(dv)):
            assert np.array_equal(dv[i]["stars"].numpy(), dv[i]["stars"].numpy()), "frozen clouds must not change"
            seen.add(dv[i]["snapshot"])
    assert seen == {91, 99}, seen
    print("PASS loader: both snapshots or one, snapshot in every item, shapes, masks, mass-balanced sampler, frozen "
          "validation clouds per snapshot")

if os.environ.get("KEEP_TMP"):                    # KEEP_TMP=1 keeps the synthetic data and datasets to look at
    print("kept", TMP)
else:
    shutil.rmtree(TMP, ignore_errors=True)
print("ALL TESTS PASSED")
