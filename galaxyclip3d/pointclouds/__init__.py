"""Build the GalaxyCLIP-3D point-cloud dataset from TNG snapshots, one snapshot at a time (numpy + h5py only, no
torch). Snapshots can be added to a dataset after the raw data of earlier ones is deleted.

Pipeline (tools/ has one script per step, each run per snapshot):
    select.py    galaxies of a snapshot that pass the filters                     -> snap_NNN/selection.h5
    lineage.py   their merger-tree branches (TNG API) and the split, shared by a galaxy's versions at all snapshots
    extract.py   each galaxy's particles within a sphere, in physical units      -> snap_NNN/shards/*.h5 (store.py)
    sampling.py  point clouds drawn from the stored particles: stars, dark matter, gas and HI

Settings live in config.py and are saved next to the output; layout.py says where every file of a dataset lives.
"""
