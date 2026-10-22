"""Stable galaxy identities shared by the 2D maps and the 3D point clouds.

Every sample in either modality should carry (snapshot, halo) so that 2D and 3D
embeddings of the same galaxy can be paired later:

* 2D (GalaxyCLIP-2D, tng50-1.2D HDF5): global sample index i belongs to block
  i // 4, and block b holds the 4 views of halo `blocks[b] = (snapshot, gid)`
  (data/block_to_gid_2d.json). The 2D gid is the FoF group index.
* 3D (tng50-1.3D domain_translation / superresolution, snapshot 99): each group
  `Subhalo_<sid>` has the attribute ParentHaloIndex = FoF group index = 2D gid.
  All 1000 subhalos in each dataset are the central (GroupFirstSub) of their halo,
  and every one of those halos has 2D maps (data/pairing_2d_3d.json).
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def load_2d_blocks(path: str | Path = DATA / "block_to_gid_2d.json") -> list[tuple[int, int]]:
    """Global 2D block index -> (snapshot, gid)."""
    return [tuple(x) for x in json.load(open(path))["blocks"]]


def halo_of_2d_sample(global_index: int, blocks: list[tuple[int, int]]) -> tuple[int, int]:
    """(snapshot, gid) of a 2D HDF5 sample (4 consecutive samples = 4 views of one halo)."""
    return blocks[int(global_index) // 4]


def load_pairing(dataset: str = "domain_translation",
                 path: str | Path = DATA / "pairing_2d_3d.json") -> list[dict]:
    """3D subhalos of one dataset with their 2D counterpart: sid, halo, central, split3d, split2d."""
    table = json.load(open(path))
    if dataset not in table:
        raise KeyError(f"{dataset!r} not in {path}; have {[k for k in table if k != 'description']}")
    return table[dataset]


def blocks_by_halo(blocks: list[tuple[int, int]], snapshot: int = 99) -> dict[int, list[int]]:
    """gid -> list of 2D block indices at one snapshot (a halo can have 2 blocks)."""
    out: dict[int, list[int]] = {}
    for b, (s, g) in enumerate(blocks):
        if s == snapshot:
            out.setdefault(g, []).append(b)
    return out
