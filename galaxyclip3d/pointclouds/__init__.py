"""Build the GalaxyCLIP-3D point-cloud dataset from a TNG snapshot (numpy + h5py only, no torch).

Pipeline (tools/ has one script per step):
    select.py    galaxies that pass the filters, split train/val/test by halo   -> selection.h5
    extract.py   each galaxy's particles within a sphere, in physical units      -> shards/*.h5 (store.py)
    sampling.py  point clouds drawn from the stored particles: stars, dark matter, gas and HI

Settings live in config.py and are saved next to the output.
"""
