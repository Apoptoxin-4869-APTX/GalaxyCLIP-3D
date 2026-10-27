# GalaxyCLIP 3D

Contrastive alignment of **3D galaxy point clouds** from the IllustrisTNG TNG50-1 simulation,
continuing [GalaxyCLIP-2D](https://github.com/Apoptoxin-4869-APTX/GalaxyCLIP-2D) (2D projected maps,
7 physical domains). The long-term goal is to align 2D maps and 3D point clouds of the **same
galaxies** in one embedding space.

## Data (ZHAW cluster)

`/cluster/projects/ska/illustris/tng50-1.3D/`, snapshot 99 (z = 0):

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
embedding file. To keep that possible, every 3D sample should carry `(snapshot, halo)`, and the
3D embeddings should follow the 2D conventions: 512-d, L2-normalized, InfoNCE with τ = 0.07.

## Raw snapshot 99 — source for new point clouds

New point clouds are to be extracted from the raw TNG50-1 data at
`/cluster/projects/ska/illustris/tng50-1/` (`snapshots/099/`, `groupcats/099/`,
`offsets/offsets_099.hdf5`). Audit from `tools/audit_snapshot.py`, 2026-10-27:

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
unreadable particles per type, 2D cube half-width, box-edge flag). Ways to complete them, both
needing a personal TNG API key (the original download job used another user's):

- per-halo API cutouts of the incomplete types — stars 368 M + dark matter 44 M particles, **~24 GB**;
- re-downloading the 109 files — **~469 GB**, completes the whole box.

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

## Code shared with GalaxyCLIP-2D

Copied from GalaxyCLIP-2D commit `8b0a377` (modality-agnostic, unchanged unless noted):

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
