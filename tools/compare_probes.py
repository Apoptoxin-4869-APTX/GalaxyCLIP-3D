"""Linear probes (and retrieval) of several cached feature sets on the galaxies they all share: e.g. a control run
whose dark-matter sphere leaves some galaxies out, against the final setup's cached embeddings.

    PYTHONPATH=pylibs:. python -s tools/compare_probes.py --data /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --features "final, stellar sphere=outputs/clip/phase_d/embeddings/c21_combined" \
                   "R_vmax sphere=outputs/clip/phase_d/embeddings_dm_rvmax/c29_dm_rvmax" \
        [--modalities dm] [--retrieval] --save outputs/clip/phase_d/dm_aperture_compare.json

Each feature set is LABEL=PREFIX, PREFIX being a cache of tools/probe_clip.py (PREFIX_train.npz, PREFIX_val.npz).
Galaxies are matched by (snapshot, subhalo); a modality counts for a galaxy only if every set has it, and only
modalities every set holds are compared (a Stage E set without HI next to a four-modality one). Probes as in
galaxyclip3d/probes.py: ridge fitted on the shared training galaxies, its strength chosen and scored on the shared
validation galaxies. --retrieval adds R@1 per modality pair on the shared validation galaxies for the sets that are
embeddings (one width for every modality).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.multimodal import MODALITIES, NEAR_DUPLICATE, masked_recall            # noqa: E402
from galaxyclip3d.probes import PROPERTIES, load_targets, lookup, ridge_probe            # noqa: E402

SPLITS = ("train", "val")


def load(prefix: str, split: str):
    with np.load(f"{prefix}_{split}.npz") as z:
        feats = {m: z[f"emb_{m}"] for m in MODALITIES if f"emb_{m}" in z}
        keys = [(int(s), int(h)) for s, h in zip(z["snapshot"], z["subhalo"])]
        return feats, keys, z["mask"].astype(bool)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the dataset's folder (for snap_099/targets.h5)")
    ap.add_argument("--features", nargs="+", required=True, help="LABEL=PREFIX of probe_clip.py caches")
    ap.add_argument("--modalities", nargs="+", default=None, choices=MODALITIES,
                    help="default: every modality all sets hold")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--save", type=Path, required=True)
    args = ap.parse_args(argv)
    sets = dict(f.split("=", 1) for f in args.features)
    data = {lab: {sp: load(pre, sp) for sp in SPLITS} for lab, pre in sets.items()}
    held = [m for m in MODALITIES if all(m in d[sp][0] for d in data.values() for sp in SPLITS)]
    if args.modalities is None:
        args.modalities = held
    elif set(args.modalities) - set(held):
        raise SystemExit(f"not every feature set holds {sorted(set(args.modalities) - set(held))}")
    shared, rows, masks = {}, {}, {}
    for sp in SPLITS:
        common = sorted(set.intersection(*(set(d[sp][1]) for d in data.values())))
        shared[sp] = common
        rows[sp] = {}
        for lab, d in data.items():
            where = {k: i for i, k in enumerate(d[sp][1])}
            rows[sp][lab] = np.array([where[k] for k in common], dtype=np.int64)
        masks[sp] = np.logical_and.reduce([data[lab][sp][2][rows[sp][lab]] for lab in data])
    if any(s != 99 for s, _ in shared["train"] + shared["val"]):
        raise SystemExit("only snapshot 99 has targets")
    targets = load_targets(args.data, 99)
    y = {sp: lookup(targets, [h for _, h in shared[sp]]) for sp in SPLITS}
    print(f"shared galaxies: train {len(shared['train']):,}, val {len(shared['val']):,} (of "
          + ", ".join(f"{lab}: {len(d['train'][1]):,} / {len(d['val'][1]):,}" for lab, d in data.items()) + ")")

    results = {lab: {} for lab in data}
    for lab, d in data.items():
        for m in args.modalities:
            mi = MODALITIES.index(m)
            results[lab][m] = {}
            for prop, (_, _, score) in PROPERTIES.items():
                ok = {sp: masks[sp][:, mi] & np.isfinite(y[sp][prop]) for sp in SPLITS}
                if min(int(o.sum()) for o in ok.values()) < 10:
                    continue
                x = {sp: d[sp][0][m][rows[sp][lab]][ok[sp]] for sp in SPLITS}
                results[lab][m][prop] = ridge_probe(x["train"], y["train"][prop][ok["train"]], x["val"],
                                                    y["val"][prop][ok["val"]], score=score)
    regress = [p for p, (_, _, s) in PROPERTIES.items() if s == "r2"]
    width = max(14, max(len(lab) for lab in data) + 2)
    for m in args.modalities:
        print(f"\n{m} features, validation R² (AUC for central, quenched)")
        print(f"{'property':18s}" + "".join(f"{lab[:width - 2]:>{width}s}" for lab in data))
        for p in PROPERTIES:
            print(f"{p:18s}" + "".join(f"{results[lab][m][p]['val']:{width}.3f}" if p in results[lab][m] else
                                       f"{'-':>{width}s}" for lab in data))
        print(f"{'mean R² (neg. = 0)':18s}" + "".join(
            f"{np.mean([max(results[lab][m][p]['val'], 0.0) for p in regress if p in results[lab][m]]):{width}.3f}"
            for lab in data))

    retrieval = {}
    if args.retrieval:
        print("\nretrieval on the shared validation galaxies, R@1 (%), mean of both directions")
        for lab, d in data.items():
            f = d["val"][0]
            if len({f[m].shape[1] for m in held}) != 1:
                continue
            emb = {m: torch.as_tensor(f[m][rows["val"][lab]]).float() for m in held}
            cols = [MODALITIES.index(m) for m in held]
            res = masked_recall(emb, torch.as_tensor(masks["val"][:, cols]), ks=(1, 10), modalities=tuple(held))
            pairs = {}
            for i, a in enumerate(held):
                for b in held[i + 1:]:
                    pairs[f"{a}-{b}"] = 50 * (res[f"r1/{a}->{b}"] + res[f"r1/{b}->{a}"])
            retrieval[lab] = {"r1_no_gas_hi": 100 * res["r1/mean_no_gas_hi"], "r10_no_gas_hi": 100 * res["r10/mean_no_gas_hi"],
                              "pairs": pairs}
            print(f"{lab:{width}s} R@1 without gas-HI {100 * res['r1/mean_no_gas_hi']:5.1f}, R@10 "
                  f"{100 * res['r10/mean_no_gas_hi']:5.1f} | " + " ".join(f"{p} {v:5.1f}" for p, v in pairs.items()))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"features": sets, "shared": {sp: len(v) for sp, v in shared.items()},
                                     "modalities": held,
                                     "near_duplicate": sorted("-".join(p) for p in NEAR_DUPLICATE),
                                     "results": results, "retrieval": retrieval}, indent=1) + "\n")
    print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
