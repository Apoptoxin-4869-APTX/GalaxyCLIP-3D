"""Choose the galaxies (Subfind subhalos) of the dataset and split them into train/val/test by halo.

Filters (from the 3D filter suggestions): SubhaloFlag = 1, at least cfg.min_particles bound star and dark-matter
particles, stellar half-mass radius at least cfg.min_rhalf_star_kpc. Gas is not required: galaxies with fewer than
cfg.min_particles gas cells stay in and get the gas modality masked later. Counts here are bound particles; the
counts inside the extracted sphere are added by extract.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config
from .snapshot import Snapshot

SPLITS = ("train", "val", "test")
BLOCKS_2D = Path(__file__).resolve().parents[2] / "data" / "block_to_gid_2d.json"
SUBHALO_FIELDS = ["SubhaloLenType", "SubhaloFlag", "SubhaloHalfmassRadType", "SubhaloMassInRadType", "SubhaloSFRinRad",
                  "SubhaloGrNr", "SubhaloPos", "SubhaloVel", "SubhaloVmax"]


def halos_with_2d_maps(snap: int, path: Path = BLOCKS_2D) -> np.ndarray:
    return np.array(sorted({g for s, g in json.load(open(path))["blocks"] if s == snap}), dtype=np.int64)


def assign_splits(group: np.ndarray, log_mstar: np.ndarray, fractions, seed: int, strata: int = 6) -> np.ndarray:
    """Split code per galaxy (0 train, 1 val, 2 test). Whole halos go to one split; within each stellar-mass stratum
    of halos (by their most massive selected galaxy) halos are dealt to the split furthest below its share."""
    groups, inverse = np.unique(group, return_inverse=True)
    size = np.bincount(inverse)
    top = np.full(len(groups), -np.inf)
    np.maximum.at(top, inverse, log_mstar)
    edges = np.quantile(top, np.linspace(0, 1, strata + 1)[1:-1])
    stratum = np.digitize(top, edges)
    rng = np.random.default_rng(seed)
    code = np.zeros(len(groups), dtype=np.int8)
    for s in range(strata):
        members = rng.permutation(np.flatnonzero(stratum == s))
        target = np.asarray(fractions, dtype=float) * size[members].sum()
        filled = np.zeros(len(fractions))
        for g in members:
            k = int(np.argmax(target - filled))
            code[g] = k
            filled[k] += size[g]
    return code[inverse]


def select(cfg: Config, snap: Snapshot) -> dict[str, np.ndarray]:
    """One entry per selected galaxy, in subhalo order."""
    cat = snap.catalog(["GroupFirstSub"], SUBHALO_FIELDS)
    u = snap.units
    n = cat["SubhaloLenType"].astype(np.int64)
    rhalf = u.kpc(cat["SubhaloHalfmassRadType"][:, 4])
    keep = ((cat["SubhaloFlag"] == 1) & (n[:, 4] >= cfg.min_particles) & (n[:, 1] >= cfg.min_particles)
            & (rhalf >= cfg.min_rhalf_star_kpc))
    ids = np.flatnonzero(keep)
    group = cat["SubhaloGrNr"][ids].astype(np.int64)
    mstar = u.msun(cat["SubhaloMassInRadType"][ids, 4])
    return dict(
        subhalo=ids.astype(np.int64),
        group=group,
        central=(cat["GroupFirstSub"][group] == ids),
        halo_has_2d=np.isin(group, halos_with_2d_maps(cfg.snap)),
        split=assign_splits(group, np.log10(np.maximum(mstar, 1.0)), cfg.split, cfg.seed),
        centre=cat["SubhaloPos"][ids].astype(np.float64),          # code units, ckpc/h
        bulk_velocity=cat["SubhaloVel"][ids].astype(np.float64),   # km/s
        vmax=cat["SubhaloVmax"][ids].astype(np.float64),           # km/s
        rhalf_star_kpc=rhalf[ids],
        mstar=mstar,                                               # Msun within 2 R1/2,*
        sfr=cat["SubhaloSFRinRad"][ids].astype(np.float64),        # Msun/yr within 2 R1/2,*
        n_gas_bound=n[ids, 0], n_dm_bound=n[ids, 1], n_stars_bound=n[ids, 4],
        offsets=snap.subhalo_offsets(ids),                         # first snapshot row per type
    )


def shard_plan(table: dict[str, np.ndarray], shard_size: int) -> np.ndarray:
    """Shard number per galaxy: galaxies sorted by particle count and dealt in turn, so shards cost about the same."""
    cost = table["n_gas_bound"] + table["n_dm_bound"] + table["n_stars_bound"]
    order = np.argsort(-cost, kind="stable")
    n_shards = int(np.ceil(len(order) / shard_size))
    shard = np.empty(len(order), dtype=np.int32)
    shard[order] = np.arange(len(order)) % n_shards
    return shard
