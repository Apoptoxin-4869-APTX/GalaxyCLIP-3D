# GalaxyCLIP-3D point clouds v1 (TNG50-1, snapshot 99)

`/cluster/projects/ska/illustris/tng50-1.pointclouds/v1/`, built 2026-11-09 from the complete snapshot copy
`/cluster/projects/ska/illustris/tng50-1.new` with the code in `galaxyclip3d/pointclouds/` and `tools/`.
Settings: `config.json` in the dataset folder (a copy of `configs/pc_v1.json`). Each shard records the code commit:
v1 says `c0c2dca+dirty` because it was built from the then-uncommitted code that the commit adding this file contains.
The history was rewritten on 2026-11-09 (commit messages only), so those hashes changed: `c0c2dca` is now `ba771e0`
and the commit adding this file is `b75d3a7`.
The snapshot folder was called `tng50-1.snap099` when v1 was built and was renamed `tng50-1.new` the same day, to hold
further snapshots (snapshot 91 next). The build settings stored inside the shards, `selection.h5` and `index.h5` still
name the old path; `config.json` and `configs/pc_v1.json` name the new one. The data are the same files.

Every selection step, from simulation subhalos to the points of a cloud, with counts, reasons and settings, is
described in **`SELECTION.md`**.

## What is in it

| File | Contents |
|---|---|
| `config.json` | the build settings |
| `selection.h5` | the 5,254 selected galaxies with their split and shard (`tools/build_selection.py`) |
| `shards/shard_0000.h5` … `shard_0082.h5` | each galaxy's particles within 10 stellar half-mass radii, 32 GB in total |
| `index.h5` | all galaxies with shard, row and particle counts at several radii |
| `feature_stats.json` | mean and spread of every feature over 1,000 training galaxies (used to standardize) |
| `frozen/val_r5.h5`, `frozen/test_r5.h5` | fixed validation and test clouds (520 and 499 galaxies) |
| `qa_clouds.pdf`, `qa.log`, `finalize.log` | quality checks and example clouds |

## Galaxies

Each object is one Subfind subhalo (galaxy), centrals and satellites alike, with the particles bound to it.

| Filter | |
|---|---|
| `SubhaloFlag = 1` | real galaxies, not clumps formed inside another galaxy |
| ≥ 4,096 bound star and dark-matter particles | twice the 2,048-point budget, so every epoch can draw different points |
| stellar half-mass radius ≥ 0.576 kpc | two gravitational softening lengths; smaller galaxies have no resolved shape |

That selects **5,254 galaxies** (3,209 centrals, 2,045 satellites) in 3,218 halos. The loader also requires 4,096 star
and dark-matter particles **inside the sphere it uses**; at the default sphere (5 stellar half-mass radii) **5,071
galaxies** qualify:

| Split | Galaxies | Centrals | Gas modality on | HI modality on | Halo has 2D maps |
|---|---|---|---|---|---|
| train | 4,052 | 2,464 | 2,756 | 3,215 | 3,671 |
| val | 520 | 337 | 394 | 472 | 469 |
| test | 499 | 319 | 347 | 422 | 457 |

Whole halos go to one split (80/10/10 of the galaxies, balanced over stellar mass, seed 4869), so a satellite never
sits in another split than its central. The gas modality is on when the sphere holds ≥ 4,096 gas cells; the HI
modality when the HI weights' effective number of cells (Σw)²/Σw² reaches 512. Galaxies without enough gas stay in
with those modalities masked; `PointCloudDataset(..., gas_required=True)` drops them instead.

## Stored particles (shards)

Positions are relative to `SubhaloPos` with the periodic box wrapped, in physical kpc; velocities are relative to
`SubhaloVel`, in km/s. Wind cells (`GFM_StellarFormationTime ≤ 0`) are removed from the stars. Above 131,072
particles per type a random subset is stored, each particle with inclusion probability π ∝ its sampling weight
(Poisson sampling: mass for stars, uniform for dark matter, half mass and half HI mass for gas), and `wcorr = 1/π`
restores unbiased mass- and HI-weighted draws.

| Type | Fields |
|---|---|
| `stars` | `pos`, `vel`, `mass`, `mass_init` [Msun], `age` [Gyr], `log_z` [log10 Z/Zsun, floored at −4], `wcorr` |
| `dm` | `pos`, `vel`, `wcorr` (all particles have the same mass, 4.5×10⁵ Msun) |
| `gas` | `pos`, `vel`, `mass`, `m_hi` = 0.76 · `NeutralHydrogenAbundance` · mass [Msun], `log_nh` [log10 cm⁻³], `log_t` [log10 K], `log_z`, `log_b` [log10 µG, floored at −6], `x_hi`, `sf` (1 if SFR > 0), `r_cell` [kpc, radius of the cell's volume], `wcorr` |

The `galaxies` table of each shard (and `index.h5`) holds per galaxy: `subhalo`, `group`, `central`, `halo_has_2d`,
`split` (0 train, 1 val, 2 test), `centre`, `bulk_velocity`, `vmax`, `rhalf_star_kpc`, `mstar`, `sfr`, bound counts,
counts inside 2, 3, 5, 7.5 and 10 stellar half-mass radii (`n_<type>_r<k>`), `hi_eff_r<k>`, and stored counts.

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
  `log_nh`, `log_t` and `log_b` are standardized with `feature_stats.json`.
- Gas and HI points are moved to a random place inside their cell's sphere, so the two clouds share no coordinates
  (checked by `tools/check_pointclouds.py`).
- Training draws are fresh every time, with one random rotation per galaxy applied to every modality's positions
  and velocities. Validation and test clouds come from `frozen/`, drawn once without rotation.
- A masked modality is returned as zeros, with `mask` = [stars, dm, gas, hi] telling which are present.
- `MassBalancedSampler` draws training galaxies evenly across six stellar-mass bins.
- Every item carries `snapshot`, `group` (= the 2D `gid`), `subhalo`, `central` and `halo_has_2d` for pairing with
  the 2D maps later; such experiments need one split common to both datasets.

```python
from torch.utils.data import DataLoader
from galaxyclip3d.dataset import MassBalancedSampler, PointCloudDataset
train = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v1", "train")
loader = DataLoader(train, batch_size=64, sampler=MassBalancedSampler(train), num_workers=8)
val = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v1", "val")   # frozen clouds
```

Speed measured on honolulu: 83 ms per training galaxy in one process, 1.1 s per batch of 32 with 4 workers.

## Caveats

- Only particles bound to the galaxy are used: other subhalos' particles and unbound gas inside the sphere are not
  in the clouds.
- Star-forming gas has the sub-grid model's effective temperature (about 10<sup>4.2</sup> K) and neutral fraction;
  the `sf` feature marks it. `NeutralHydrogenAbundance` in star-forming gas includes molecular hydrogen.
- Positions and velocities are scaled per galaxy, so absolute size and mass are not in the clouds; `rhalf_star_kpc`,
  `vmax` and `mstar` in the index can be added as galaxy-level inputs if needed.
- Frozen sets exist for the 5-radii sphere only; for another radius run
  `tools/finalize_pointclouds.py --radius 3` (radii available: 2, 3, 5, 7.5, 10).

## Rebuilding

```bash
PYTHONPATH=pylibs python3 -s tools/build_selection.py --config configs/pc_v1.json      # 7 s
env -u SLURM_JOB_ID sbatch --array=0-41%16 slurm/build_pointclouds.submit             # 14 min
PYTHONPATH=pylibs python3 -s tools/finalize_pointclouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v1
python tools/check_pointclouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v1 --pdf qa_clouds.pdf
```

`pylibs/` holds numpy and h5py for `/usr/bin/python3` (`pip install --target pylibs --python-version 3.12
--only-binary=:all: numpy h5py`), which works on both cpu nodes. Two builds with the same settings give identical
files. Tests on a synthetic snapshot: `PYTHONPATH=pylibs python3 -s tests/test_pointclouds.py` (the loader part needs
torch).
