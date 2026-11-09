# GalaxyCLIP 3D

Contrastive alignment of **3D galaxy point clouds** from the IllustrisTNG TNG50-1 simulation,
continuing [GalaxyCLIP-2D](https://github.com/Apoptoxin-4869-APTX/GalaxyCLIP-2D) (2D projected maps,
7 physical domains). The long-term goal is to align 2D maps and 3D point clouds of the **same
galaxies** in one embedding space.

## Data (ZHAW cluster)

`/cluster/projects/ska/illustris/tng50-1.3D/`, snapshot 99 (z = 0). These are the 3D sets prepared before this
project, 1,000 central galaxies each; our own dataset, built from the raw snapshot, is the point-cloud dataset v3
below.

| Set | Points per particle type | Subhalos (train / val / test) |
|---|---|---|
| `domain_translation/` | 8,192 | 900 / 50 / 50 |
| `superresolution/` | 16,384 | 900 / 50 / 50 |

Each HDF5 group `Subhalo_<SubhaloID>` holds `PartType0` (gas), `PartType1` (dark matter) and
`PartType4` (stars) with coordinates and physical fields, plus ~50 subhalo catalog attributes
(`ParentHaloIndex`, `AssignedSplit`, masses, metallicities, ...). Masses are already rescaled
(`MassScaling`), coordinates are relative to the subhalo.

**Caveats found so far**
- The two sets contain 992 common subhalos but split them differently (802 in the same split).
- With 50 validation subhalos, one retrieval hit is 2 % R@1 — use repeated splits or
  cross-validation for stable numbers.

## Pairing with the 2D maps

All 1,000 subhalos of each 3D set are the **central galaxy** (`GroupFirstSub`) of one of the
~1,000 most massive halos at z = 0, and every one of those halos has 2D maps (4 views each) in
GalaxyCLIP-2D. The link is `ParentHaloIndex` (3D) = FoF group index `gid` (2D):

- `data/pairing_2d_3d.json` — per 3D subhalo: `sid`, `halo`, `central`, `split3d`, `split2d`
- `data/block_to_gid_2d.json` — 2D HDF5 sample order: block `b` (samples `4b..4b+3`) = halo
  `(snapshot, gid)`. Note: each 2D file holds the next 1,000 *existing* gids from its start, so
  `gid = chunk_start + position // 4` is wrong for ~47 % of samples — use this table.
- `galaxyclip3d/ids.py` — helpers to look these up.

**The 2D and 3D splits disagree:** of the 1,000 paired galaxies (domain_translation), only 4
are validation in both; 88 3D-training galaxies are 2D-validation and 46 3D-validation galaxies
are 2D-training. Any 2D↔3D experiment needs one common split (e.g. the 2D split restricted to
these 1,000 halos), otherwise one side has seen the "held-out" galaxies.

### Planned route to 2D↔3D alignment
Keep the trained 2D model frozen, export its embeddings keyed by `(snapshot, gid, view)`, and
train the 3D encoder to match them (anchor / ImageBind-style). The repos then only share an
embedding file. For that, every item of the point-cloud dataset v3 carries `snapshot`, `group` (= the 2D `gid` at
that snapshot) and `subhalo`, and the 3D embeddings should follow the 2D conventions: 512-d, L2-normalized, InfoNCE
with τ = 0.07.

## Raw snapshots — source for new point clouds

**Complete copy (the one to use):** `/cluster/projects/ska/illustris/tng50-1.new/` holds snapshot 99 complete, laid
out as `snapshots/099/` (680 files, 2.92 TB), `groupcats/099/` (680 files) and `offsets/offsets_099.hdf5`; the folder
can hold several snapshots side by side. It was downloaded on 2026-11-02 with our own `tools/download_snapshot.py`
(SLURM job 160622, 25 files in parallel, 5 h 44 min; 731 refused requests retried, none failed) and checked on
2026-11-09: all 1,361 files present with the server's sizes, 9 files re-hashed against TNG's checksums, and
`tools/audit_snapshot.py --root` finds no unreadable particle and all 2,991 halos with 2D maps complete.

```bash
env -u SLURM_JOB_ID sbatch slurm/download_snapshot.submit --snap N                       # download snapshot N
python3 tools/download_snapshot.py status --dest /cluster/projects/ska/illustris/tng50-1.new --snap N --watch 10
```

Downloading needs a personal TNG API key in `~/.tng_api_key` (mode 600). The TNG data server often refuses valid
requests for about a minute (HTTP 403 or 504); the downloader retries them.

**Storage limit.** `/cluster/projects/ska` has a **10 TiB quota shared by the whole ska group**. `df` shows the 175 TB
filesystem instead, and so does the downloader's free-space check, which therefore cannot see the quota. A full
snapshot is 2.93 TB. On 2026-11-09 a download of snapshot 91 filled the quota at 29% (850 GB), which blocked writes
for everyone in the group until it was deleted. Usage afterwards: 10.145 of 10.995 TB (851 GB free), of which
`tng50-1.new` 2.93 TB, the older copy 2.47 TB, `tng50-1.2D` 1.92 TB, `tng50-1.3D` 1.63 TB, the point clouds
0.05 TB and other ska data 1.15 TB. To see what is really free:

```bash
python3 -c "import os; d='/cluster/projects/ska'; q=int(os.getxattr(d,'ceph.quota.max_bytes')); u=int(os.getxattr(d,'ceph.dir.rbytes')); print(f'{(q-u)/1e9:,.0f} GB free')"
```

### The older copy (incomplete)

`/cluster/projects/ska/illustris/tng50-1/` (`snapshots/099/`, `groupcats/099/`, `offsets/offsets_099.hdf5`) is
another user's download, which this project leaves untouched. Audit from `tools/audit_snapshot.py`, 2026-10-27:

**The snapshot is incomplete.** 108 of its 680 files were never downloaded and
`snap_099.657.hdf5` is truncated, so 109 files (~16 % of every particle type) are unreadable.
Particles are stored in FoF-group order, so the gaps remove whole blocks of specific halos:

| Halos with 2D maps (snapshot 99) | Complete on disk |
|---|---|
| gid 0–99 | 55 / 100 |
| gid 100–499 | 289 / 400 |
| gid 500–999 | 210 / 499 |
| gid 1000–1999 | 176 / 992 |
| gid 2000–3014 | 0 / 1,000 |
| **All** | **730 / 2,991** |

Gas is complete for all of them; stars are incomplete for 1,975 halos, dark matter for 244 and
black holes for 504. Per-halo table: `data/snap099_halo_availability.csv` (FoF particle counts,
unreadable particles per type, 2D cube half-width, box-edge flag). The complete copy above replaced it as the source
of new point clouds, and the point-cloud build reads that one; `tools/audit_snapshot.py` and
`tools/check_2d_selection.py` still default to this older copy (both take `--root`).

### How the 2D maps select particles

The 2D maps were made with [skais-mapper](https://github.com/phdenzel/skais-mapper) (configs
`tng50-1-{snapshot}-2D-{gid range}`). Each map keeps the **FoF-group particles inside a cube
centred on the central subhalo's `SubhaloPos`, with half-width = the central's gas half-mass
radius** (`SubhaloHalfmassRadType[central, 0]`, the map's `extent` attribute), rotated by
`rotxy` for views 1–3. `tools/check_2d_selection.py` reproduces the stored `num_particles`
of the unrotated view:

| gid | gas | dark matter | stars |
|---|---|---|---|
| 100 | exact | −8 of 1.89 M | −1 of 1.40 M |
| 500 | exact | +1 | exact |
| 1031 | −2 | −4 | exact |
| 2913 | exact | +2 | not on disk |
| 765 | −1 | −2 | exact |

The differences are particles on a cube face that flip with floating-point rounding; the exact
arithmetic of the July 2025 run is not recoverable from the current skais-mapper code.

Worth knowing before matching 2D content in 3D:
- **No periodic wrapping.** 24 of the 2,991 cubes cross the box edge and lose whatever lies
  beyond it (gid 765: 96,454 gas cells in the 2D map, 126,175 with wrapping). Flagged in the CSV.
- **Star maps include wind-phase cells** (no `GFM_StellarFormationTime > 0` cut).
- `temp` and `bfield` are mass-weighted means along the line of sight; `hi-21cm` converts the
  HI column to a brightness temperature in mK.
- SPH smoothing radii use (0.75·π·V)^(1/3), ~2.1× the equal-volume radius (affects only the 2D blur).
- skais-mapper's `halo_index` is "the i-th halo with a central" for the subhalo and "FoF group i"
  for the particles; the two agree only while every halo up to i has a central (true for these gids).
- Halos whose central has < 10,000 gas cells were skipped and replaced by the next gid, hence
  "the next 1,000 existing gids" per 2D file.

### Reading the files

The official loader `illustris_python` expects `snapdir_099/`, `groups_099/` and
`../postprocessing/offsets/`, so with this layout it needs a directory of symlinks or patched
paths (skais-mapper reads it as is). `FileOffsets/SnapByType` is stored as (files, 6), not
[6, files] as the TNG documentation says. Coordinates are float64 and the largest halo has
415 M dark-matter particles, so read big halos in chunks (or `loadSubset(..., float32=True)`).

```bash
python tools/audit_snapshot.py --csv data/snap099_halo_availability.csv   # ~2 min, numpy + h5py
python tools/check_2d_selection.py 100 500 1031 2913 765                  # ~1.5 min
```

## Point-cloud dataset v3 (our own, 2026-11-09)

`/cluster/projects/ska/illustris/tng50-1.pointclouds/v3/`, built one snapshot at a time; it holds snapshot 99, built
from the complete snapshot copy: 5,254 galaxies (centrals and satellites) with four modalities, namely stars, dark
matter, gas and HI, 2,048 points each, drawn fresh every epoch by `galaxyclip3d/dataset.py`. At the default sphere of
5 stellar half-mass radii, 5,071 galaxies qualify (train 4,052 / val 520 / test 499). Up to about 262,144 (2¹⁸)
particles are stored per type and galaxy. Every galaxy records its snapshot and redshift, and further snapshots can be
added after the raw data of earlier ones is deleted. Splits follow galaxies through time: a galaxy's versions at
different snapshots share the split of the z = 0 halo they end in (SubLink merger trees from the TNG API). v3 is v2
(which replaced v1 the same day) rebuilt with this code; snapshot 99's content is unchanged, and v2 was deleted.
Fields, units, masks, adding a snapshot and caveats: **`DATASET.md`**. How galaxies, particles and points are
selected, step by step with counts: **`SELECTION.md`**.

| Step, per snapshot | Code |
|---|---|
| select galaxies, merger-tree branches, split units | `tools/build_selection.py` (`galaxyclip3d/pointclouds/select.py`, `lineage.py`, `tng_api.py`) |
| extract particles into shards | `slurm/build_pointclouds.submit` → `tools/build_pointclouds.py` (`extract.py`, `store.py`) |
| indexes, feature statistics, frozen val/test clouds | `tools/finalize_pointclouds.py` |
| quality checks and example clouds | `tools/check_pointclouds.py` |
| draw clouds, PyTorch loader | `galaxyclip3d/pointclouds/sampling.py`, `galaxyclip3d/dataset.py` |

Where every file lives: `galaxyclip3d/pointclouds/layout.py`. The build needs only numpy and h5py (`pylibs/`, see
`DATASET.md`); the loader needs torch. `tests/test_pointclouds.py` runs the whole pipeline on two synthetic snapshots,
adding the second to a dataset built from the first.

**Adding a snapshot.** For snapshot S: download it, run `tools/build_selection.py --snap S` (about 8 minutes, mostly
fetching merger-tree branches from the TNG API), the SLURM shard job with `SNAP=S` (10–25 minutes), then
`tools/finalize_pointclouds.py --snap S` and `tools/check_pointclouds.py --snap S`, and delete the raw snapshot;
the commands are in `DATASET.md`. Each snapshot adds about 47 GB of point clouds. A raw snapshot needs 2.93 TB, more
than the 851 GB free in ska, so the next one fits only after snapshot 99's raw copy is deleted.

**Masked modalities.** Gas and HI are switched off for a galaxy whose sphere holds too little gas: gas below 4,096
cells, HI below 512 effective cells. At 5 radii that is 1,574 of the 5,071 usable galaxies for gas (31%) and 962 for
HI (19%). 678 (13%) have no gas at all, 626 of them satellites whose gas their host stripped; small galaxies are hit
most (gas off for 44% below 10⁹ solar masses, 5% above 10¹¹). The loader returns a masked cloud as zeros with its
`mask` entry false, so the loss can skip it; `gas_required=True` drops these galaxies instead (3,497 left). The QA PDF
draws clouds without this rule, so it says "nothing to draw" only for galaxies with no gas at all.

**Storage cap.** 916 galaxies (17%) have more than 262,144 particles of one type inside the stored sphere (stars 697,
dark matter 762, gas 321). For them a random subset is stored: each particle is kept with a probability proportional
to its draw weight (mass for stars, equal for dark matter, half mass and half HI mass for gas), and `wcorr` = 1 / that
probability keeps the draws unbiased (`SELECTION.md`, step 2.5). The number kept varies by a few hundred around the
cap, and the thinning ignores position, so the centre is thinned as much as the outskirts. `tools/cap_study.py`
measured what this costs on 60 capped galaxies read in full: at 5 radii the median error is 0.04–0.10 of the spread
between two clouds of the same galaxy (at most about 1% extra noise), the 90th percentile at most 0.14; at 2 radii it
reaches 0.08–0.17 (90th percentile up to 0.32 for gas). A cap of 2¹⁹ would take 59.5 GB instead of 46.6 GB for
little gain.

| Analysis tool | Shows | Pictures (local, `outputs/` is not in git) |
|---|---|---|
| `tools/plot_clouds.py` | a galaxy's four clouds in 3D, coloured by age, speed, temperature and HI rotation | `outputs/pointcloud_examples/` |
| `tools/compare_radii.py` | clouds drawn at several sphere sizes (e.g. 5, 10, 15 radii; beyond 10 read from the raw snapshot) | `outputs/radius_comparison/` |
| `tools/cap_study.py` | particles per galaxy, and what each storage cap from 2¹² to 2²³ costs in size and accuracy | `outputs/cap_study/` |
| `tools/check_pointclouds.py --pdf` | one drawn cloud per modality for 12 galaxies across stellar mass | `snap_NNN/qa_clouds.pdf` in the dataset |

## Code shared with GalaxyCLIP-2D

Copied from GalaxyCLIP-2D commit `12d8db8` (modality-agnostic, unchanged unless noted):

| Module | Contents |
|---|---|
| `galaxyclip3d/losses.py` | symmetric InfoNCE (pairwise, anchor, all-pairs with optional queue), in-batch retrieval accuracy |
| `galaxyclip3d/queue.py` | memory queue of extra negatives — the 2D model's queue, moved into a standalone class |
| `galaxyclip3d/training.py` | ZClip adaptive gradient clipping, full-gallery R@k, progress printer, queue ramp schedule, `ModelEMA` (new) |
| `galaxyclip3d/utils.py` | DDP helpers, SIGUSR1 requeue handling, atomic checkpoints, stable W&B run ids |
| `slurm/train.submit.template` | multi-GPU job with requeue and data staging (cluster rules in the header) |

Lessons from the 2D runs worth keeping: full-gallery R@1 (not batch-level accuracy) is the
metric; grow the memory queue gradually (a single large jump collapsed retrieval); averaging
the weights of two checkpoints added +0.9 R@1 (`ModelEMA` is the in-training version, not yet
tested on a full run); test-time augmentation must apply the same
transform to query and gallery.

`python tests/test_shared.py` checks the copied modules (needs torch; run on a GPU node's venv).

## License
AGPL-3.0, as GalaxyCLIP-2D (see `LICENSE`).
