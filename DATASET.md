# GalaxyCLIP-3D point clouds v3 (TNG50-1, built one snapshot at a time)

`/cluster/projects/ska/illustris/tng50-1.pointclouds/v3/`, settings in `config.json` there (a copy of
`configs/pc_v3.json` without the snapshot). It holds snapshot 99 (z = 0), built 2026-11-09 from the complete copy
`/cluster/projects/ska/illustris/tng50-1.new` with the code in `galaxyclip3d/pointclouds/` and `tools/`; more
snapshots can be added later, each after the raw data of the previous one has been deleted.

**v3 and v2.** v3 is v2 rebuilt with code that records each galaxy's snapshot and can add snapshots: the snapshot-99
galaxies, splits, shard plan, index columns, frozen clouds, feature statistics and all 2,075 stored particle arrays
(46.2 GB) are identical to v2's, checked array by array before v2 was deleted (2026-11-09). What is new is the folder
layout below, the `snapshot`, `redshift` and merger-tree columns, and the split rule that keeps a galaxy's versions at
different snapshots in one split. v2 itself had replaced v1 the same day by doubling the storage cap.

Every selection step, from simulation subhalos to the points of a cloud, with counts, reasons and settings, is
described in **`SELECTION.md`**.

## What is in it

| File | Contents |
|---|---|
| `config.json` | settings shared by all snapshots (`snap` is null here) |
| `index.h5` | every galaxy of every snapshot, with its snapshot, split, shard, row and particle counts |
| `snapshots.json` | per snapshot: redshift, galaxies, usable galaxies, split sizes, build date, code commit, source |
| `splits.h5` | the split of every split unit, shared by all snapshots, and snapshot 99's `SubhaloGrNr` (see Splits) |
| `feature_stats.json` | mean and spread of every feature over 1,000 training galaxies (used to standardize; fixed) |
| `snap_099/config.json` | the full settings of snapshot 99's build |
| `snap_099/selection.h5` | its 5,254 selected galaxies with split and shard (`tools/build_selection.py`) |
| `snap_099/lineage.h5` | each galaxy's SubLink main branch (snapshot, subhalo) from the TNG API |
| `snap_099/shards/shard_0000.h5` … `shard_0082.h5` | each galaxy's particles within 10 stellar half-mass radii, 46.6 GB |
| `snap_099/index.h5` | snapshot 99's part of `index.h5` |
| `snap_099/frozen/val_r5.h5`, `test_r5.h5` | fixed validation and test clouds (520 and 499 galaxies) |
| `snap_099/qa_clouds.pdf`, `qa.log` | quality checks and example clouds |

A snapshot added later gets its own `snap_NNN/` folder with the same files.

## Galaxies

Each object is one Subfind subhalo (galaxy) at one snapshot, centrals and satellites alike, with the particles bound
to it. The same filters apply at every snapshot:

| Filter | |
|---|---|
| `SubhaloFlag = 1` | real galaxies, not clumps formed inside another galaxy |
| ≥ 4,096 bound star and dark-matter particles | twice the 2,048-point budget, so every epoch can draw different points |
| stellar half-mass radius ≥ 0.576 kpc | two gravitational softening lengths at z ≤ 1 (snapshot ≥ 50); smaller galaxies have no resolved shape |

At snapshot 99 that selects **5,254 galaxies** (3,209 centrals, 2,045 satellites) in 3,218 halos. The loader also
requires 4,096 star and dark-matter particles **inside the sphere it uses**; at the default sphere (5 stellar
half-mass radii) **5,071 galaxies** qualify:

| Split | Galaxies | Centrals | Gas modality on | HI modality on | Halo has 2D maps |
|---|---|---|---|---|---|
| train | 4,052 | 2,464 | 2,756 | 3,215 | 3,671 |
| val | 520 | 337 | 394 | 472 | 469 |
| test | 499 | 319 | 347 | 422 | 457 |

The gas modality is on when the sphere holds ≥ 4,096 gas cells; the HI modality when the HI weights' effective number
of cells (Σw)²/Σw² reaches 512. Galaxies without enough gas stay in with those modalities masked;
`PointCloudDataset(..., gas_required=True)` drops them instead.

## Splits across snapshots

A galaxy at snapshot 91 and its descendant at snapshot 99 look alike. If they sat in different splits, the test set
would hold galaxies the model has in effect trained on. So splits are given to **split units** shared by all
snapshots: the FoF halo at snapshot 99 (z = 0) in which the galaxy's SubLink main descendant branch ends
(`galaxyclip3d/pointclouds/lineage.py`). Every version of a galaxy, and every galaxy that ends up in the same z = 0
halo, gets that halo's split.

- At snapshot 99 a galaxy is its own descendant, so its unit is its halo: exactly v2's rule (80/10/10 of the galaxies,
  whole halos, balanced over stellar mass, seed 4869). Snapshot 99's splits are v2's.
- For another snapshot, `tools/build_selection.py` fetches each galaxy's main branch from the TNG API
  (`sublink/simple.json`, about 0.8 kB each; 5,254 branches took 7.6 minutes with 8 parallel requests and none failed)
  and looks up the halo of its snapshot-99 descendant in `splits.h5`, so the raw data of snapshot 99 is not needed.
- Units already in `splits.h5` keep their split. New ones, z = 0 halos without a selected galaxy so far, are dealt
  like the first snapshot's halos, with seed (4869, snapshot), and recorded.
- A galaxy whose branch does not reach snapshot 99 uses its own halo at its own snapshot as unit.
- Consequence: at an earlier snapshot, a halo can hold galaxies of different splits when they end in different z = 0
  halos (a satellite that later leaves its host, for example). `build_selection.py` and `check_pointclouds.py`
  report how many.

The branches also allow pairing a galaxy with its own progenitors or descendants in other snapshots: every galaxy
carries `z0_subhalo`, and `lineage.h5` lists its main branch back to the earliest snapshot (all 5,254 snapshot-99
galaxies have a progenitor at snapshot 91).

## Stored particles (shards)

Positions are relative to `SubhaloPos` with the periodic box wrapped, in physical kpc; velocities are relative to
`SubhaloVel`, in km/s; both use the snapshot's own scale factor. Stellar ages count from the snapshot's own time. Wind
cells (`GFM_StellarFormationTime ≤ 0`) are removed from the stars. Above about 262,144 particles per type a random
subset is stored, each particle with inclusion probability π ∝ its sampling weight (Poisson sampling: mass for stars,
uniform for dark matter, half mass and half HI mass for gas), and `wcorr = 1/π` restores unbiased mass- and
HI-weighted draws; the stored number varies by a few hundred around the cap. At snapshot 99 the cap applies to 697
galaxies for stars, 762 for dark matter and 321 for gas; 1.12 billion particles are stored (stars 359 million, dark
matter 481 million, gas 278 million).

| Type | Fields |
|---|---|
| `stars` | `pos`, `vel`, `mass`, `mass_init` [Msun], `age` [Gyr], `log_z` [log10 Z/Zsun, floored at −4], `wcorr` |
| `dm` | `pos`, `vel`, `wcorr` (all particles have the same mass, 4.5×10⁵ Msun) |
| `gas` | `pos`, `vel`, `mass`, `m_hi` = 0.76 · `NeutralHydrogenAbundance` · mass [Msun], `log_nh` [log10 cm⁻³], `log_t` [log10 K], `log_z`, `log_b` [log10 µG, floored at −6], `x_hi`, `sf` (1 if SFR > 0), `r_cell` [kpc, radius of the cell's volume], `wcorr` |

Each shard's file attributes record its `snapshot`, `redshift`, `source` folder, settings, code commit, host and
date. Snapshot 99's shards record `e21837a+dirty`: they were built with the multi-snapshot code before it was
committed (the same commit string v2 recorded for its own uncommitted cap change). Its `galaxies` table (and `index.h5`) holds per galaxy: `snapshot`, `redshift`, `subhalo`, `group` (FoF halo at
that snapshot), `central`, `halo_has_2d`, `split` (0 train, 1 val, 2 test), `unit_snap` and `unit_group` (its split
unit), `z0_subhalo` and `z0_group` (its snapshot-99 descendant and halo, −1 if none), `has_branch`, `centre`,
`bulk_velocity`, `vmax`, `rhalf_star_kpc`, `mstar`, `sfr`, bound counts, counts inside 2, 3, 5, 7.5 and 10 stellar
half-mass radii (`n_<type>_r<k>`), `hi_eff_r<k>`, and stored counts. A galaxy is identified by (`snapshot`,
`subhalo`); subhalo numbers repeat across snapshots.

## The four modalities (what the loader returns)

`galaxyclip3d/dataset.py` draws 2,048 points per modality from a galaxy's particles inside the sphere
(`galaxyclip3d/pointclouds/sampling.py`):

| Modality | Drawn from | Weighted by | Features per point |
|---|---|---|---|
| `stars` | star particles | mass (without replacement) | x y z vx vy vz age log_z |
| `dm` | dark-matter particles | uniform (without replacement) | x y z vx vy vz |
| `gas` | gas cells | mass (without replacement) | x y z vx vy vz log_nh log_t log_z log_b x_hi sf |
| `hi` | gas cells | neutral-H mass (with replacement) | x y z vx vy vz x_hi |

- Positions are divided by the sphere radius and velocities by the galaxy's `SubhaloVmax`; `age`, `log_z`,
  `log_nh`, `log_t` and `log_b` are standardized with `feature_stats.json`, computed once from snapshot 99's training
  galaxies and kept when snapshots are added (`finalize_pointclouds.py --recompute-stats` redoes it over all).
- Gas and HI points are moved to a random place inside their cell's sphere, so the two clouds share no coordinates
  (checked by `tools/check_pointclouds.py`).
- Training draws are fresh every time, with one random rotation per galaxy applied to every modality's positions
  and velocities. Validation and test clouds come from each snapshot's `frozen/`, drawn once without rotation.
- A masked modality is returned as zeros, with `mask` = [stars, dm, gas, hi] telling which are present.
- `MassBalancedSampler` draws training galaxies evenly across six stellar-mass bins.
- Every item carries `snapshot`, `redshift`, `group` (= the 2D `gid` at that snapshot), `subhalo`, `central`,
  `halo_has_2d` and `z0_subhalo`, for pairing with the 2D maps or across snapshots; 2D↔3D experiments need one split
  common to both datasets.

```python
from torch.utils.data import DataLoader
from galaxyclip3d.dataset import MassBalancedSampler, PointCloudDataset
root = "/cluster/projects/ska/illustris/tng50-1.pointclouds/v3"
train = PointCloudDataset(root, "train")                      # every snapshot in the dataset
loader = DataLoader(train, batch_size=64, sampler=MassBalancedSampler(train), num_workers=8)
val = PointCloudDataset(root, "val", snapshots=[99])          # frozen clouds of chosen snapshots
```

Speed measured on honolulu (v2, same files): 94 ms per training galaxy in one process.

## Adding a snapshot

Each step reads only what it needs, so the raw snapshot can be deleted after step 3. Replace `S` by the snapshot
number (folder `snapshots/0SS` etc. in the copy).

```bash
env -u SLURM_JOB_ID sbatch slurm/download_snapshot.submit --snap S                                       # 2.93 TB, about 5 h
PYTHONPATH=pylibs python3 -s tools/build_selection.py --config configs/pc_v3.json --snap S                 # ~8 min (TNG API)
SNAP=S env -u SLURM_JOB_ID sbatch --export=ALL --array=0-41%16 slurm/build_pointclouds.submit               # ~25 min
PYTHONPATH=pylibs python3 -s tools/finalize_pointclouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 --snap S
python tools/check_pointclouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 --snap S --pdf .../v3/snap_0SS/qa_clouds.pdf
```

- **Space.** One full snapshot is 2.93 TB plus 7 GB of catalogs, and `/cluster/projects/ska` has a 10 TiB quota shared
  by the whole group (`df` does not show it; when it is full, nobody in the group can write there). With snapshot
  99's raw copy (2.92 TB) still
  there, about 0.85 TB was free on 2026-11-09, so the next snapshot fits only after that copy is deleted. Each
  snapshot adds about 47 GB of point clouds.
- **What must exist before deleting the raw data:** the snapshot's `snap_NNN/` with all shards, `index.h5` and
  `frozen/`, and a clean `check_pointclouds.py`. Linking later snapshots needs only `splits.h5` and the TNG API.
  Deleting `tng50-1.new/snapshots/0SS` frees the 2.93 TB; the group catalog (`groupcats/0SS`, 7 GB) and offsets
  (1 GB) are small enough to keep, and with them a selection can be recomputed later. Tools that read raw particles
  (`cap_study.py`, `compare_radii.py` beyond 10 radii, `audit_snapshot.py`) stop working for a deleted snapshot.
- **Settings.** `build_selection.py` refuses a snapshot whose settings differ from the dataset's (except `root`,
  `snap` and `chunk_rows`), so every snapshot is built the same way. It also refuses to replace an existing,
  different selection; delete `snap_NNN/` to rebuild one snapshot.
- **Order.** Snapshots can be added in any order. One at a time: `build_selection.py` locks `splits.h5`
  (`.splits.lock`).
- **API.** The key must be in `~/.tng_api_key` (mode 600). Failed requests are retried; branches already fetched are
  cached in `snap_NNN/lineage.h5`, so rerunning after an API outage fetches only the rest.

## Caveats

- Only particles bound to the galaxy are used: other subhalos' particles and unbound gas inside the sphere are not
  in the clouds.
- Star-forming gas has the sub-grid model's effective temperature (about 10<sup>4.2</sup> K) and neutral fraction;
  the `sf` feature marks it. `NeutralHydrogenAbundance` in star-forming gas includes molecular hydrogen.
- Positions and velocities are scaled per galaxy, so absolute size and mass are not in the clouds; `rhalf_star_kpc`,
  `vmax`, `mstar` and `redshift` in the index can be added as galaxy-level inputs if needed.
- Frozen sets exist for the 5-radii sphere only; for another radius run
  `tools/finalize_pointclouds.py --snap S --radius 3` (radii available: 2, 3, 5, 7.5, 10).
- Frozen clouds are seeded by (4869, subhalo, 1), as in v2; the subhalo number repeats across snapshots, which only
  reuses a random stream for different particles.
- The size filter of 0.576 kpc equals two softening lengths only at z ≤ 1; at higher redshift it is stricter than that
  (`build_selection.py` prints a note).

## Rebuilding snapshot 99

```bash
PYTHONPATH=pylibs python3 -s tools/build_selection.py --config configs/pc_v3.json                          # 8 min
SNAP=99 env -u SLURM_JOB_ID sbatch --export=ALL --array=0-41%16 slurm/build_pointclouds.submit             # 25 min
PYTHONPATH=pylibs python3 -s tools/finalize_pointclouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 --snap 99
python tools/check_pointclouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 --snap 99 --pdf .../v3/snap_099/qa_clouds.pdf
```

`pylibs/` holds numpy and h5py for `/usr/bin/python3` (`pip install --target pylibs --python-version 3.12
--only-binary=:all: numpy h5py`), which works on both cpu nodes. Two builds with the same settings give identical
files. Tests on two synthetic snapshots, including adding the second to a dataset built from the first:
`PYTHONPATH=pylibs python3 -s tests/test_pointclouds.py` (the loader part needs torch).
