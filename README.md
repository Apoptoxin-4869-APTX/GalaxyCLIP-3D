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

## Contrastive alignment of the four modalities (started 2026-11-13)

Goal: one encoder per modality (stars, dark matter, gas, HI) so that a galaxy's four clouds land close together in a
shared 512-d space, as GalaxyCLIP-2D did for its 7 maps. The metric is full-gallery recall R@1/5/10 per modality
pair over the 520 validation galaxies, each pair over the galaxies that have both modalities (galleries: 520 with
dark matter, 394 with gas, 472 with HI), reported as a mean over the 12 directions with and without gas↔HI (both are
drawn from the same gas cells, like 2D's hi/hi-21cm). The 499 test galaxies are kept for the final numbers.

**Orientation.** Every modality gets its own random rotation (and a mirror image half of the time); see `DATASET.md`.
With one shared orientation, matching only the direction of angular momentum, no learning, already finds the right
galaxy for 35% (stars→gas), 31% (stars→HI) and 10% (stars→dark matter) of the validation galaxies.

**Baselines without a network** (`tools/clip_baselines.py`, validation set, chance 0.23%):

| Baseline | R@1, mean over pairs | without gas↔HI |
|---|---|---|
| angular-momentum direction, clouds in a shared frame | 29.9% | 19.0% |
| angular-momentum direction, independent rotations | 0.3% | 0.2% |
| about 20 orientation-free statistics per cloud + CCA per pair, fitted on the training set | 15.2% | 4.0% |

| Piece | Code |
|---|---|
| masked all-pairs InfoNCE, masked full-gallery recall (2D functions untouched) | `galaxyclip3d/multimodal.py` |
| encoders: PointNet (1.2 M parameters per modality), patch transformer à la Point-MAE (11.9 M), PointNet++ and PointNeXt (2.2 and 2.3 M at the Stage E settings) | `galaxyclip3d/models/` |
| training (one GPU, bf16, AdamW + cosine, ZClip, optional EMA, requeue, W&B optional) | `galaxyclip3d/train_clip.py`, `configs/clip_*.json` |
| job: copies the dataset (v3, or `SOURCE_DATA`) to the node's local disk, excludes trinity, fresko and irvine | `slurm/train_clip.submit` |
| properties for probes (halo mass, SFR, metallicities, colour, stellar age, HI mass) | `tools/build_targets.py` → `snap_099/targets.h5` |
| tests (orientation, masked loss and recall, encoders, ablation options, principal axes, Vector Neurons, a short synthetic training run, scoring and probe helpers, Stage E settings) | `tests/test_clip.py` |

```bash
env -u SLURM_JOB_ID sbatch slurm/train_clip.submit configs/clip_pointnet.json                    # PointNet baseline
env -u SLURM_JOB_ID sbatch slurm/train_clip.submit configs/clip_patch.json --set lr=3e-4 batch_size=512
```

The trainer evaluates the untrained model first (epoch 0), then every `eval_every` epochs; `control=shuffle_pairs`
re-pairs galaxies at random as a leak test (validation must stay at chance) and `rotation=shared` measures the
orientation shortcut. Loader speed measured on honolulu: 454 ms per galaxy on a first read from shared storage, 27 ms
from memory, 175 galaxies/s with 8 workers, hence the copy to local disk.

Ablation settings for Phase C, all off by default:

| Setting | Effect |
|---|---|
| `points=1024` | points per cloud; validation needs frozen sets with that count (`finalize_pointclouds.py --budget 1024 --frozen-only`) |
| `common_radii=[3,5,7.5]` | trains and validates only on galaxies usable at every listed sphere, so sphere sizes compare on the same galaxies |
| `inputs=no_velocities` or `no_scalars` | the encoders see positions plus the physical values, or positions plus velocities |
| `point_invariants=true` | adds each point's distance from the centre, speed, radial velocity and tangential speed, which rotation and mirroring leave unchanged (the distance only when velocities are dropped) |
| `loss_pairs=stars_anchor` | trains only the three stars pairs; validation still covers all 12 directions |
| `shared_encoder=true` | one PointNet for all modalities, each with its own first layer and projection head |
| `patch_centres=radial` | patch centres picked in radially equalised positions, so they follow the density |
| `dm_aperture=vmax` | cuts the dark matter at the galaxy's R_vmax and scales it by R_vmax instead of the stellar sphere, so the dark-matter cloud no longer carries the stars' size (a control for the probes); needs `r_vmax_kpc` in `targets.h5`, keeps the ~80% of galaxies whose R_vmax lies inside the stored particles with at least 4,096 dark-matter particles, and draws seeded validation clouds instead of the frozen ones |

**Scoring finished runs (Phase D).** The test set is scored once, after the final setup is fixed; validation is used
for every choice before that.

| Piece | Code |
|---|---|
| reload a run, embed a split in one pass for several models, retrieval scores as during training | `galaxyclip3d/evaluation.py` |
| linear probes: ridge per property, strength chosen on validation; R² (AUC for central and quenched) | `galaxyclip3d/probes.py` |
| retrieval scores of runs on val or test, mean and spread over seeds, `--rotation` to override | `tools/eval_clip.py` |
| probes on each modality's embedding, next to hand-made statistics and the untrained network | `tools/probe_clip.py` |
| the no-network baselines on the test set | `tools/clip_baselines.py --split test` |
| thesis figures (PDF + PNG) and tables (LaTeX + Markdown) from the saved results, with bootstrap intervals over test galaxies | `tools/clip_report.py` → `outputs/clip/report/` |
| pictures of a run's latent space on the validation galaxies (t-SNE maps by modality, property and redshift; cross-modal similarity) | `tools/latent_space.py` → `<run>/latent/` |
| how well a property (default the mean stellar age) can be read off frozen embeddings: linear, nearest-neighbour and MLP probes trained on the training galaxies, scored on validation, next to hand-made statistics, an untrained network and the stellar mass | `tools/age_probe.py` |
| probes and retrieval of several cached feature sets on the galaxies they share (e.g. a run with another dark-matter sphere against the final setup) | `tools/compare_probes.py` |

```bash
V3=/cluster/projects/ska/illustris/tng50-1.pointclouds/v3
PYTHONPATH=pylibs:. python -s tools/eval_clip.py --data $V3 --split test --runs outputs/clip/phase_c/c21_combined \
    outputs/clip/phase_c/c22_combined_seed2 outputs/clip/phase_c/c28_combined_seed3 --save outputs/clip/phase_d/test_final.json
PYTHONPATH=pylibs:. python -s tools/probe_clip.py --data $V3 --runs outputs/clip/phase_c/c21_combined \
    --untrained outputs/clip/phase_c/c21_combined --statistics --eval-split val \
    --cache outputs/clip/phase_d/embeddings --save outputs/clip/phase_d/probes_val.json
PYTHONPATH=pylibs:. python -s tools/clip_report.py          # reads outputs/clip, writes outputs/clip/report
```

Probe features of the training galaxies come from one fixed draw per galaxy (`PointCloudDataset(..., draw_seed=...)`,
seeded by draw seed, snapshot and subhalo); validation and test galaxies use their frozen clouds. Property floors from
`build_targets.py` (no gas, no black hole, no star formation) are left out of the fits; "quenched" (no star formation)
is probed as a yes/no property instead.

**Stage E settings (2026-11-18).** Stage E leaves out HI for now and uses larger clouds of larger galaxies.
`configs/clip_stage_e.json` is the final setup (C21) with three changes:

| Setting | Stage E | Effect |
|---|---|---|
| `modalities` | `["stars", "dm", "gas"]` | one encoder per listed modality; the loss uses their 3 pairs, and validation R@1 (which picks best.pt) is the mean over their 6 directions |
| `min_particles` | `8192` | at least 8,192 star and 8,192 dark-matter particles inside the sphere, and 8,192 gas cells for the gas modality (else masked): 3,036 / 406 / 367 galaxies instead of 4,052 / 520 / 499 (gas on for 2,056 / 306 / 262). Only values above the dataset's 4,096 work |
| `points` | `4096` | points per cloud; the frozen 4,096-point validation and test sets exist |

The frozen validation clouds of the remaining galaxies are the ones every earlier run was scored on. For a
like-for-like reference, `tools/eval_clip.py --min-particles 8192 --modalities stars dm gas` scores a four-modality
run on the same galaxies without HI: the final setup (trained with HI, scored at its own 2,048 points) reaches
**42.4 ± 0.4% R@1** over the 6 directions on the 406 validation galaxies (3 seeds; stars–DM 22.9%, stars–gas 86.9%,
DM–gas 17.4%; `outputs/clip/stage_e/val_reference_final_no_hi.json`).

`configs/clip_stage_e_pointnet2.json` is the same with PointNet++ encoders (`encoder=pointnet2`), to test whether
local structure helps: two set-abstraction levels of 512 and 128 centres, each grouped with its 32 nearest points,
then a PointNet over the 128 centres (`pointnet_widths`). Centres come from farthest-point sampling in radially
equalised positions (`sa_sampling=radial`): on real star clouds that puts 46% of 512 centres inside the radius holding
half the points, plain farthest-point sampling 1%. 7.4 M parameters in all, against 6.8 M for the Stage E PointNet.

`configs/clip_stage_e_pointnext.json` uses PointNeXt encoders (`encoder=pointnext`, Qian et al. 2022): a stem layer,
then four stages, each a set-abstraction block followed by residual InvResMLP blocks (PointNeXt-B's pattern 1, 2, 1, 1;
blocks widen 4x and back). The 4,096 points are reduced to 1,024, 512, 256 and 128 centres with 32 nearest points per
group; widths 64, 128, 256, 256 (capped so an encoder has 2.3 M parameters, 7.6 M in all). Adapted like the PointNet++:
radially equalised centres, nearest points with positions divided by the group radius and its logarithm kept,
LayerNorm after the max. On the CPU a forward and backward pass takes about 3 times as long as the PointNet's.

`configs/clip_stage_e_principal.json` is the Stage E PointNet with every cloud turned into its own principal axes
before its encoder (`principal_axes=true`): x along the longest axis of the positions' second moments, z along the
shortest (stars and gas are drawn by mass, so these are the mass-weighted axes). The sign of each axis is arbitrary, so
training draws one of the 8 sign choices per cloud and evaluation averages the embeddings over all 8. The encoders then
no longer have to learn from the random rotations that orientation does not matter: on 16 validation galaxies, turning
every modality on its own changed the embeddings by at most 2e-6, and `rotation=shared` and `independent` give the
same input. How well the axes are defined (300 training galaxies, two independent 4,096-point draws each, medians): the
short axis comes out 1.2° (stars), 0.9° (gas) and 6.1° (dark matter) apart between draws, the long axis 11.6°, 3.6° and
14.8°; half of the stellar clouds and most dark-matter clouds are nearly round in the plane of their two long axes
(b/a > 0.95), so their turn within that plane stays close to random. No extra parameters; evaluation runs each encoder
8 times.

`configs/clip_stage_e_vn.json` uses Vector Neurons PointNet encoders (`encoder=vn_pointnet`; Deng et al. 2021), which no
rotation or mirroring of a cloud can change, with no frame to estimate: round clouds, whose principal axes are loose,
are no special case. Each point's position and velocity are two 3D vectors and its other values, plus its distance,
speed, radial and tangential velocity, are scalars. Four layers of 32, 64, 128 and 256 vector channels (`vn_widths`) and
64, 128, 256 and 512 scalar channels (`vn_scalar_widths`): vectors are mixed with the same weights for x, y and z, their
lengths feed the scalars, the Vector Neurons LeakyReLU acts along learned directions, and a sigmoid of the scalars
scales the vectors. Pooling is a softmax-weighted mean along a learned direction (Vector Neurons' hard max gives that
direction no gradient) and the plain mean, read off as dot products with three vectors computed from them. 8.3 M
parameters in all (6.8 M for the PointNet). On 16 validation galaxies, turning every modality on its own changed the
embeddings by at most 1e-7 (float32), in training and in evaluation, with a single pass. The many vector-sized
intermediate results took 2.9 times the PointNet's memory, so `vn_checkpoint=true` recomputes each layer in the backward
pass: 1.07 times the PointNet's memory, and a CPU training step 2.7 times as long.

Runs from e08 on set `zclip_warmup=20`: ZClip only measures the gradient norm for its first `zclip_warmup` steps, and
the default 100 steps cover the first 9 epochs at Stage E's 11 steps per epoch, the whole learning-rate warmup.

```bash
env -u SLURM_JOB_ID sbatch slurm/train_clip.submit configs/clip_stage_e.json \
    --set run_name=e01_baseline out_dir=outputs/clip/stage_e/e01_baseline
PYTHONPATH=pylibs:. python -s tools/eval_clip.py --data $V3 --split val --runs outputs/clip/phase_c/c21_combined \
    --min-particles 8192 --modalities stars dm gas --save outputs/clip/stage_e/val_reference_c21.json
```

Stage E on all six snapshots of `tng50-1.pointclouds/complete` trains on 17,480 galaxies with at least 8,192
particles (5.8 times v3's 3,036) and, with `val_snapshots=[99]`, validates on the same 406 galaxies as the v3 runs.
The job copies 279 GB, and the training galaxies' particles take 204 GB, so it asks for more memory and CPUs:

```bash
SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=28 \
    env -u SLURM_JOB_ID sbatch --cpus-per-task=32 --mem=280G slurm/train_clip.submit configs/clip_stage_e.json \
    --set run_name=e18_complete_independent out_dir=outputs/clip/stage_e/e18_complete_independent \
    rotation=independent zclip_warmup=20 'val_snapshots=[99]'
```

The Stage E PointNet peaks at 58.7 GiB of GPU memory at batch 256, more than a 40 GB A100 holds.
`pointnet_checkpoint=true` recomputes each PointNet layer in the backward pass instead of storing its intermediate
results: the same losses and gradients (the tests compare them), 27.7 GiB at batch 256, and a training step about a
quarter longer (A100, batch 128: 0.284 s instead of 0.229 s; batch 256 with the option: 0.56 s). e19 runs this way on
one of sacramento's A100s: the copy takes about 10 minutes there, and an epoch 56 s (30% of it waiting for data), so
600 epochs take about 9.3 h:

```bash
SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=44 \
    env -u SLURM_JOB_ID sbatch --nodelist=sacramento --exclude=fresko,irvine --gres=gpu:a100sxm:1 \
    --cpus-per-task=48 --mem=300G --time=16:00:00 slurm/train_clip.submit configs/clip_stage_e.json \
    --set run_name=e19_complete_shared out_dir=outputs/clip/stage_e/e19_complete_shared \
    rotation=shared zclip_warmup=20 'val_snapshots=[99]' pointnet_checkpoint=true
```

e20 and e21 train the patch transformer on the same data with the architecture of C20, the best patch run of phase C:
128 patches of 32 points around centres that follow the density, width 192, depth 4, 3 heads (2.4 M parameters per
particle type), and the patch runs' learning rate of 5e-4. e20 rotates the particle types independently:

```bash
SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=28 \
    env -u SLURM_JOB_ID sbatch --job-name=e20_complete_patch_independent --nodelist=trinity \
    --exclude=fresko,irvine --gres=gpu:h200sxm:1 --cpus-per-task=32 --mem=280G --time=12:00:00 \
    slurm/train_clip.submit configs/clip_stage_e.json \
    --set run_name=e20_complete_patch_independent out_dir=outputs/clip/stage_e/e20_complete_patch_independent \
    rotation=independent encoder=patch width=192 depth=4 heads=3 patches=128 patch_size=32 patch_centres=radial \
    lr=5e-4 zclip_warmup=20 'val_snapshots=[99]'
```

e21 is the same command with `e21_complete_patch_shared` as job name, run name and folder, and `rotation=shared`.

e22 is e17 (Vector Neurons) on the complete dataset. It needs 62.5 GB of GPU memory, more than an A100 holds, and a
step takes 1.09 s on an H200, six times PointNet's 0.18 s, so 600 epochs take about 13.5 h there. It runs on sanjose's
H100 (94 GB) with 44 loader workers, because data loading on sanjose was about three times slower per worker (e12):

```bash
SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=44 \
    env -u SLURM_JOB_ID sbatch --job-name=e22_complete_vn --nodelist=sanjose --exclude=fresko,irvine \
    --gres=gpu:h100pcie:1 --cpus-per-task=48 --mem=300G --time=24:00:00 \
    slurm/train_clip.submit configs/clip_stage_e_vn.json \
    --set run_name=e22_complete_vn out_dir=outputs/clip/stage_e/e22_complete_vn rotation=independent zclip_warmup=20 \
    'val_snapshots=[99]'
```

e23 and e24 repeat e19 and e21 with the stars' age hidden from the stellar encoder, `drop_inputs=["stars.age"]`. Any
per-point physical value can be hidden this way ("modality.column"); positions and velocities cannot, and the data stay
as they are. Both run on sacramento's A100s; e24's patch network peaks at 34.7 GiB of the card's 39.5, so it runs with
expandable memory segments:

```bash
SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=44 \
    env -u SLURM_JOB_ID sbatch --job-name=e23_complete_shared_noage --nodelist=sacramento --exclude=fresko,irvine \
    --gres=gpu:a100sxm:1 --cpus-per-task=48 --mem=300G --time=16:00:00 \
    slurm/train_clip.submit configs/clip_stage_e.json \
    --set run_name=e23_complete_shared_noage out_dir=outputs/clip/stage_e/e23_complete_shared_noage rotation=shared \
    zclip_warmup=20 'val_snapshots=[99]' pointnet_checkpoint=true 'drop_inputs=["stars.age"]'
SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=44 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    env -u SLURM_JOB_ID sbatch --job-name=e24_complete_patch_shared_noage --nodelist=sacramento \
    --exclude=fresko,irvine --gres=gpu:a100sxm:1 --cpus-per-task=48 --mem=200G --time=16:00:00 \
    slurm/train_clip.submit configs/clip_stage_e.json \
    --set run_name=e24_complete_patch_shared_noage out_dir=outputs/clip/stage_e/e24_complete_patch_shared_noage \
    rotation=shared encoder=patch width=192 depth=4 heads=3 patches=128 patch_size=32 patch_centres=radial lr=5e-4 \
    zclip_warmup=20 'val_snapshots=[99]' 'drop_inputs=["stars.age"]'
```

e25 and e26 repeat e09 (PointNet++) and e11 (PointNeXt), both with shared rotations, on the complete dataset. They need
52.8 and 41.8 GiB of GPU memory, more than an A100 holds, so they run on trinity:

```bash
for run in "e25_complete_pointnet2_shared pointnet2 12" "e26_complete_pointnext_shared pointnext 16"; do
    set -- $run
    SOURCE_DATA=/cluster/projects/ska/illustris/tng50-1.pointclouds/complete WORKERS=28 \
        env -u SLURM_JOB_ID sbatch --job-name=$1 --nodelist=trinity --exclude=fresko,irvine --gres=gpu:h200sxm:1 \
        --cpus-per-task=32 --mem=280G --time=$3:00:00 slurm/train_clip.submit configs/clip_stage_e_$2.json \
        --set run_name=$1 out_dir=outputs/clip/stage_e/$1 rotation=shared zclip_warmup=20 'val_snapshots=[99]'
done
```

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
