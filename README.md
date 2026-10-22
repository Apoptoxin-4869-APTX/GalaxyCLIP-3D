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
