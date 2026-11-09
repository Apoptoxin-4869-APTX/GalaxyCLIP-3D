# How the point-cloud dataset selects its galaxies, particles and points

This describes every selection step of the GalaxyCLIP-3D point-cloud dataset v1, with the counts after each step,
the reason for it, the setting that controls it and the code that does it. Field descriptions and loading are in
`DATASET.md`.

- **Source:** TNG50-1 snapshot 99 (z = 0), complete copy at `/cluster/projects/ska/illustris/tng50-1.snap099`.
- **Dataset:** `/cluster/projects/ska/illustris/tng50-1.pointclouds/v1`.
- **Settings:** `configs/pc_v1.json`, copied to `config.json` in the dataset folder.

Selection happens at three levels:

| Level | From | To | Code |
|---|---|---|---|
| Galaxies | 5,688,113 Subfind subhalos | 5,254 selected; 5,071 usable at the default sphere | `pointclouds/select.py`, `dataset.py` |
| Particles | 20.1 billion gas, dark-matter and star particles | 0.81 billion stored (34 GB) | `pointclouds/extract.py` |
| Points | a galaxy's stored particles | 4 clouds × 2,048 points per draw | `pointclouds/sampling.py` |

## 1. Galaxies

### 1.1 What counts as a galaxy

One object is one **Subfind subhalo**, a gravitationally bound galaxy, whether it is the central galaxy of its
friends-of-friends (FoF) halo or a satellite. Its particles are those Subfind assigned to it. Particles of other
subhalos, and unbound particles, are never included, even if they lie inside its sphere.

The 2D maps used a different object: all FoF particles inside a cube around the central galaxy, with half-width equal
to the central's gas half-mass radius. Applied to every halo, that rule gives cubes smaller than 1 kpc (13 halos),
cubes wider than the halo itself (193), stellar bodies larger than the cube (164), and cubes cut by the box edge (97).
It also cannot represent satellites, so the 3D dataset does not use it.

### 1.2 Filters

Applied to the group catalog, in this order (`select()` in `galaxyclip3d/pointclouds/select.py`):

| Step | Rule | Galaxies left | Centrals | Satellites | Why | Setting |
|---|---|---|---|---|---|---|
| 0 | all Subfind subhalos | 5,688,113 | 4,462,345 | 1,225,768 | | |
| 1 | `SubhaloFlag = 1` | 5,686,764 | 4,462,340 | 1,224,424 | flag 0 marks clumps that formed inside another galaxy, not real galaxies | |
| 2 | ≥ 4,096 bound star particles | 6,037 | 3,538 | 2,499 | twice the 2,048-point budget, so each epoch can draw a different subset; with fewer, clouds repeat points and the repetition reveals the galaxy's mass | `min_particles` |
| 3 | ≥ 4,096 bound dark-matter particles | 5,729 | 3,535 | 2,194 | same reason | `min_particles` |
| 4 | stellar half-mass radius ≥ 0.576 kpc | **5,254** | **3,209** | **2,045** | two gravitational softening lengths (0.288 kpc at z = 0); below that a galaxy's shape is set by the simulation's numerics | `min_rhalf_star_kpc` |

Gas is deliberately not required here: 1,056 of the 5,254 have fewer than 4,096 bound gas cells, mostly satellites
whose gas their host stripped. They stay in, with the gas and HI modalities masked (section 1.5). The 5,254 galaxies
belong to 3,218 halos. 4,721 of them sit in a halo that has 2D maps, 2,678 of those as its central.

### 1.3 Train, validation and test split

Whole FoF halos go to one split, so a satellite is never in another split than its central and the galaxies around
it (`assign_splits()`). Halos are grouped into 6 strata by the stellar mass of their most massive selected galaxy. Within each stratum, in a
random order (seed 4869, `seed`), each halo goes to the split furthest below its target share of 80/10/10 of the
galaxies (`split`). Result: train 4,198, validation 536, test 520 galaxies.

### 1.4 Usable galaxies at the chosen sphere

Steps 2 and 3 count all bound particles. Training uses only the particles inside a sphere around the galaxy (section
3), so the loader also requires **≥ 4,096 star particles and ≥ 4,096 dark-matter particles inside that sphere**
(`galaxy_masks()` in `galaxyclip3d/dataset.py`). The shards store counts at five radii, in units of each galaxy's
stellar half-mass radius:

| Sphere | Usable | Train | Val | Test | Centrals | Gas modality on | HI modality on |
|---|---|---|---|---|---|---|---|
| 2 radii | 3,981 | 3,174 | 418 | 389 | 2,518 | 2,130 | 3,178 |
| 3 radii | 4,672 | 3,744 | 474 | 454 | 2,896 | 2,856 | 3,747 |
| **5 radii (default)** | **5,071** | **4,052** | **520** | **499** | **3,120** | **3,497** | **4,109** |
| 7.5 radii | 5,182 | 4,141 | 531 | 510 | 3,172 | 3,843 | 4,239 |
| 10 radii | 5,222 | 4,172 | 534 | 516 | 3,191 | 4,009 | 4,284 |

### 1.5 Modality masks

Stars and dark matter are always present in a usable galaxy. The other two depend on the sphere:

- **Gas** is on when the sphere holds ≥ 4,096 gas cells (`min_particles`).
- **HI** is on when the effective number of cells carrying the neutral hydrogen, (Σw)² / Σw² with w the HI mass per
  cell, reaches 512 (`hi_min_eff`). This rejects galaxies whose neutral gas sits in a handful of cells.

A masked modality is returned as an all-zero cloud with its `mask` entry false, and should be left out of the loss.
`PointCloudDataset(..., gas_required=True)` drops galaxies without gas instead of masking them.

## 2. Particles stored for each galaxy

`extract()` in `galaxyclip3d/pointclouds/extract.py`, run per shard by `tools/build_pointclouds.py`:

1. **Bound particles.** The offsets file gives each subhalo's first row per particle type (`Subhalo/SnapByType`) and
   the catalog its length (`SubhaloLenType`); in the snapshot these rows are contiguous. Gas (PartType0), dark matter
   (PartType1) and stars (PartType4) are read in chunks of 8 million rows (`chunk_rows`). Tracers (PartType3) and
   black holes (PartType5) are not used.
2. **Centre and units.** Positions are taken relative to `SubhaloPos`, using the nearest periodic image so galaxies at the box edge stay whole, then converted to physical kpc. Velocities are converted to km/s (multiplied by √a) and taken relative to `SubhaloVel`.
3. **Sphere.** Particles within 10 stellar half-mass radii are kept (`extract_radius`). Counts inside 2, 3, 5, 7.5
   and 10 radii are recorded (`radii`), plus the HI effective cell count at each.
4. **Wind cells.** PartType4 entries with `GFM_StellarFormationTime ≤ 0` are wind-phase gas, not stars, and are
   removed. They are a median 0.9% of the star-particle mass in halos with 2D maps, at most 6%.
5. **Cap.** If more than 131,072 particles of one type lie in the sphere (`cap`), a random subset is kept. Each
   particle is kept with probability π = min(1, c·q), with c chosen so the expected number kept is 131,072. Here q is
   the particle's mass for stars, the same for every dark-matter particle, and half its mass plus half its HI mass for gas. The stored weight
   `wcorr` = 1/π makes mass- and HI-weighted draws unbiased again, so each original particle's chance of being drawn
   stays proportional to its mass (or HI mass). The random seed is (4869, subhalo, type), so rebuilds are identical.
   Capped galaxies: 1,040 for stars, 1,346 for dark matter, 725 for gas. Median stored per galaxy: 21,925 star
   particles, 51,930 dark-matter particles, 19,708 gas cells.
6. **Fields.** Only what the modalities need is kept, in float32:

| Type | Stored field | From the snapshot |
|---|---|---|
| all | `pos` [kpc] | `Coordinates` − `SubhaloPos`, periodic box wrapped, × a/h |
| all | `vel` [km/s] | `Velocities` × √a − `SubhaloVel` |
| all | `wcorr` | 1 / probability of being kept by the cap (1 when not capped) |
| stars | `mass`, `mass_init` [Msun] | `Masses`, `GFM_InitialMass` × 10¹⁰/h |
| stars | `age` [Gyr] | flat ΛCDM time between `GFM_StellarFormationTime` and the snapshot, clipped at 0 |
| stars, gas | `log_z` | log₁₀(`GFM_Metallicity` / 0.0127), floored at −4 |
| gas | `mass` [Msun] | `Masses` × 10¹⁰/h |
| gas | `m_hi` [Msun] | 0.76 × `NeutralHydrogenAbundance` × mass |
| gas | `log_nh` [log₁₀ cm⁻³] | 0.76 × `Density` / proton mass |
| gas | `log_t` [log₁₀ K] | from `InternalEnergy` and `ElectronAbundance`: T = (γ−1) u μ m_p / k_B, μ = 4 / (1 + 3X_H + 4X_H x_e) |
| gas | `log_b` [log₁₀ µG] | \|`MagneticField`\| × h/a² × 2.6×10⁻⁶ G, floored at 10⁻⁶ µG |
| gas | `x_hi` | `NeutralHydrogenAbundance` |
| gas | `sf` | 1 if `StarFormationRate` > 0 |
| gas | `r_cell` [kpc] | radius of a sphere with the cell's volume, (3 m / 4πρ)^(1/3) |

Not stored: particle IDs, gravitational potential, the abundances of 10 individual elements, tagged metals, stellar
photometry, birth positions and velocities, cooling and AGN-radiation rates, energy dissipation, Mach number,
magnetic divergence, `CenterOfMass`, `InternalEnergyOld` and the halo finder's local density estimates. They remain
available in the snapshot if a future modality needs them.

### Particle counts and sizes through these steps

| | Gas | Dark matter | Stars |
|---|---|---|---|
| Whole snapshot | 8.74 billion (1.85 TB) | 10.08 billion (0.64 TB) | 1.29 billion, incl. wind (0.26 TB) |
| Bound to the 5,254 galaxies | 1.75 billion (20%) | 3.91 billion (39%) | 1.25 billion (97%) |
| Inside 10 radii (stars: wind removed) | 0.48 billion (5.5%) | 1.30 billion (12.9%) | 1.20 billion (93%) |
| Stored after the cap | 0.213 billion (2.4%) | 0.351 billion (3.5%) | 0.249 billion (19%) |
| Bytes per particle, snapshot → stored | 212 → 62 | 64 → 28 | 204 → 45 |
| Stored size | 13.2 GB | 9.9 GB | 11.1 GB |

Tracers add 0.16 TB to the snapshot's 2.93 TB and are not used.

## 3. Points drawn for each cloud

`draw()` in `galaxyclip3d/pointclouds/sampling.py`, called by the loader for every training item:

1. **Sphere.** Only stored particles within 5 stellar half-mass radii are candidates (`train_radius`; 2, 3, 7.5 or 10 can be chosen without rebuilding).
2. **Points per cloud.** 2,048 (`budget`).
3. **Weights and replacement.**

   | Cloud | Drawn from | Probability proportional to | Replacement |
   |---|---|---|---|
   | stars | star particles | mass × `wcorr` | without |
   | dm | dark-matter particles | `wcorr` | without |
   | gas | gas cells | mass × `wcorr` | without |
   | hi | gas cells | `m_hi` × `wcorr` | with: a few cells can hold most of the neutral gas |

   Draws without replacement use Efraimidis–Spirakis keys, a standard method for weighted sampling without replacement.
4. **Jitter.** Each gas and HI point is moved to a random place inside its cell's sphere (`r_cell`), so the two
   clouds, both drawn from gas cells, never share coordinates.
5. **Scaling.** Positions are divided by the sphere radius, so every cloud fills a sphere of radius 1. Velocities are divided by
   the galaxy's `SubhaloVmax`. Age, metallicity, density, temperature and field strength are standardized with the
   mean and spread of 1,000 training galaxies (`feature_stats.json`).
6. **Rotation.** Training clouds get one random rotation per galaxy, the same for all four modalities. Validation and test clouds are
   drawn once, unrotated (`frozen/`, seed (4869, subhalo, 1); 520 and 499 galaxies at 5 radii).
   *Caveat:* because a galaxy's four clouds share an orientation, a contrastive model can match them by orientation
   instead of physics. Independent rotations per modality are not implemented yet.

Each training draw is new, so over the epochs a large galaxy is seen through many different subsets of its particles.

## 4. Settings and what changing them needs

| Setting | v1 value | Controls | After a change, rerun |
|---|---|---|---|
| `min_particles` | 4,096 | galaxy filter (steps 2–3), usability inside the sphere, gas mask | everything (selection changes) |
| `min_rhalf_star_kpc` | 0.576 | size filter (step 4) | everything |
| `split`, `seed` | 80/10/10, 4869 | split by halo; cap and frozen-set seeds | everything |
| `extract_radius`, `radii` | 10; 2, 3, 5, 7.5, 10 | stored sphere and recorded counts | extraction and finalize |
| `cap` | 131,072 | particles stored per type and galaxy | extraction and finalize |
| `train_radius` | 5 | default sphere for drawing | finalize (frozen sets) only |
| `budget` | 2,048 | points per cloud | finalize (frozen sets) only |
| `hi_min_eff` | 512 | HI mask | nothing to rebuild; the loader reads it |

"Everything" is `tools/build_selection.py`, the SLURM extraction job and `tools/finalize_pointclouds.py` (see the
rebuilding section of `DATASET.md`). A different value needs a new output folder: the selection script refuses to
mix settings in one folder.
