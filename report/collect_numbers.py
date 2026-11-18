"""Collect every number the project report and the supervisor deck quote into one file, read from saved results, so
the two documents cannot disagree.

    cd GalaxyCLIP_3D && PYTHONPATH=~/tools/report_pylibs:pylibs:. python -s report/collect_numbers.py
    # -> outputs/project_report/numbers.json

Sources: the thesis tables in outputs/clip/report/report.md (built by tools/clip_report.py from the run outputs), the
Phase D files in outputs/clip/phase_d/, the dataset's index and targets, and the final run's config. Facts that live
only in the documentation (download times, storage, literature) are under "facts" with their source.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from galaxyclip3d.dataset import dm_vmax_radius, galaxy_masks          # noqa: E402
from galaxyclip3d.pointclouds.config import Config                    # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table                # noqa: E402

CLIP = REPO / "outputs/clip"
D = CLIP / "phase_d"
OUT = REPO / "outputs/project_report"
V3 = Path("/cluster/projects/ska/illustris/tng50-1.pointclouds/v3")
SPLITS = ("train", "val", "test")
MODS = ("stars", "dm", "gas", "hi")
PAIRS = ("stars-dm", "stars-gas", "stars-hi", "dm-gas", "dm-hi", "gas-hi")


def load(p: Path):
    return json.loads(Path(p).read_text())


def md_tables(path: Path) -> dict:
    """{name: {caption, header, rows, note}} for every table of report.md, keyed by its tables/<name>.tex file."""
    lines = path.read_text().splitlines()
    out, i = {}, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("**") and s.endswith("**") and i + 2 < len(lines) and lines[i + 2].startswith("|"):
            caption, j, rows = s.strip("*"), i + 2, []
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            note, name = "", None
            while j < len(lines):
                t = lines[j].strip()
                if t.startswith("*") and t.endswith("*") and not t.startswith("**"):
                    note = t.strip("*")
                m = re.match(r"\(`tables/(\w+)\.tex`\)", t)
                if m:
                    name = m.group(1)
                    break
                j += 1
            out[name] = {"caption": caption, "header": rows[0], "rows": rows[2:], "note": note}
            i = j
        i += 1
    return out


def mean_dirs(res: dict, with_gas_hi: bool) -> float:
    """Mean R@1 (%) over search directions, as tools/clip_report.py: 10 without gas<->HI, 12 with."""
    v = [res[f"r1/{a}->{b}"] for a in MODS for b in MODS if a != b
         and (with_gas_hi or {a, b} != {"gas", "hi"}) and f"r1/{a}->{b}" in res]
    return 100 * float(np.mean(v))


def dataset_numbers() -> dict:
    cfg = Config.load(V3 / "config.json")
    idx = read_table(V3 / "snap_099/index.h5")
    usable, mask = galaxy_masks(idx, cfg, 5.0)
    logm = np.log10(idx["mstar"][usable])
    per = {}
    for i, sp in enumerate(SPLITS):
        k = usable & (idx["split"] == i)
        per[sp] = {"galaxies": int(k.sum()), "centrals": int((k & (idx["central"] > 0)).sum()),
                   "gas_on": int((k & mask[:, 2]).sum()), "hi_on": int((k & mask[:, 3]).sum()),
                   "halo_has_2d": int((k & (idx["halo_has_2d"] > 0)).sum())}
    snaps = load(V3 / "snapshots.json")["99"]
    return {
        "snapshot": 99, "redshift": 0.0, "selected": int(len(idx["subhalo"])),
        "selected_centrals": int((idx["central"] > 0).sum()),
        "selected_satellites": int((idx["central"] == 0).sum()),
        "halos": int(len(np.unique(idx["group"]))), "usable_r5": int(usable.sum()),
        "splits": per, "built": snaps["built"],
        "gas_off_r5": int((usable & ~mask[:, 2]).sum()), "hi_off_r5": int((usable & ~mask[:, 3]).sum()),
        "log_mstar_usable": {"min": float(logm.min()), "median": float(np.median(logm)), "max": float(logm.max())},
        "settings": {"min_particles": cfg.min_particles, "budget": cfg.budget, "extract_radius": cfg.extract_radius,
                     "train_radius": cfg.train_radius, "radii": list(cfg.radii), "seed": cfg.seed},
    }


def rvmax_subset_numbers() -> dict:
    """The galaxies the dark-matter frame control keeps (R_vmax sphere drawable), per split, and why others drop."""
    cfg = Config.load(V3 / "config.json")
    idx = read_table(V3 / "snap_099/index.h5")
    usable, _ = galaxy_masks(idx, cfg, 5.0)
    keep = usable & np.isfinite(dm_vmax_radius(V3, idx, cfg))
    t = read_table(V3 / "snap_099/targets.h5")
    where = {int(s): i for i, s in enumerate(t["subhalo"])}
    k = np.array([where.get(int(s), -1) for s in idx["subhalo"]])
    r = np.where(k >= 0, t["r_vmax_kpc"][np.maximum(k, 0)], np.nan)
    n = np.where(k >= 0, t["n_dm_rvmax"][np.maximum(k, 0)], np.nan)
    beyond = usable & ~(r <= cfg.extract_radius * idx["rhalf_star_kpc"])
    sparse = usable & ~beyond & ~(n >= cfg.min_particles)
    logm = np.log10(idx["mstar"])
    return {"splits": {sp: int((keep & (idx["split"] == i)).sum()) for i, sp in enumerate(SPLITS)},
            "kept": int(keep.sum()), "usable": int(usable.sum()), "beyond_extent": int(beyond.sum()),
            "too_few_dm": int(sparse.sum()),
            "median_log_mstar_kept": float(np.median(logm[keep])), "median_log_mstar_all": float(np.median(logm[usable]))}


def compare_numbers(path: Path) -> dict | None:
    """Probes (DM embedding and means) and retrieval of a compare_probes.py file."""
    if not path.exists():
        return None
    d = load(path)
    regress = None
    out = {"shared": d["shared"], "features": d["features"], "retrieval": d["retrieval"], "probes": {}}
    for lab, res in d["results"].items():
        out["probes"][lab] = {}
        for m, props in res.items():
            vals = {p: v["val"] for p, v in props.items()}
            regress = regress or [p for p in vals if p not in ("central", "quenched")]
            out["probes"][lab][m] = {"props": vals,
                                     "mean_r2": float(np.mean([max(vals[p], 0.0) for p in regress if p in vals]))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = md_tables(CLIP / "report/report.md")
    summary = load(CLIP / "report/summary.json")
    ts = load(D / "test_summary.json")
    base_test = load(D / "baselines_test.json")
    c21 = load(CLIP / "phase_c/c21_combined/config.json")
    hist = load(CLIP / "phase_c/c21_combined/history.json")
    n = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset_numbers(),
        "rvmax_subset": rvmax_subset_numbers(),
        "tables": tables,
        "test": {"r1_final": summary["test_r1_final"], "interval": summary["test_interval"],
                 "r1_base": summary["test_r1_base"], "final_minus_base": summary["final_minus_base"],
                 "diff_interval": summary["diff_interval"]},
        "test_summary_keys": sorted(ts),
        "baselines_test": {k: {"r1_10dir": mean_dirs(v, False), "r1_12dir": mean_dirs(v, True)}
                           for k, v in base_test.items()},
        "final_config": {k: c21.get(k) for k in ("encoder", "pointnet_widths", "embed_dim", "encoder_dim",
                                                 "temperature", "learn_temperature", "batch_size", "epochs", "lr",
                                                 "min_lr_ratio", "weight_decay", "warmup_epochs", "radius", "rotation",
                                                 "mirror", "loss_pairs", "seed", "eval_every")},
        "final_train": {"final_temperature": hist["temperature"][-1], "final_loss": hist["train_loss"][-1],
                        "gpu_peak_gb": max(hist.get("gpu_peak_gb") or [0])},
        "curves": {k: [[v["epoch"], round(100 * v["r1/mean_no_gas_hi"], 2)]
                       for v in load(CLIP / f"{run}/history.json")["val"] if v]
                   for k, run in (("pointnet_b", "pointnet_baseline"), ("patch_b", "patch_main"),
                                  ("c21", "phase_c/c21_combined"))},
        "dm_control": compare_numbers(D / "dm_aperture_compare.json"),
        "c30": compare_numbers(D / "dm_subset_compare.json"),
        "facts": {
            "baselines_val": {"source": "README.md, tools/clip_baselines.py on validation (chance 0.23%)",
                              "spin_shared": [29.9, 19.0], "spin_independent": [0.3, 0.2], "stats_cca": [15.2, 4.0]},
            "raw_snapshot": {"source": "README.md, memory notes",
                             "size_tb": 2.92, "files": 680, "download": "2026-11-02, 5 h 44 min, 25 files in parallel",
                             "refused_retried": 731, "old_copy_missing_files": 109,
                             "old_copy_complete_2d_halos": "730 of 2,991"},
            "storage": {"source": "DATASET.md / README.md", "particles_billion": 1.12, "stars_m": 359, "dm_m": 481,
                        "gas_m": 278, "shards_gb": 46.6, "cap": 262144, "capped_galaxies": 916, "capped_stars": 697,
                        "capped_dm": 762, "capped_gas": 321, "cap_cost": "median 0.04-0.10 of the draw-to-draw spread at 5 radii"},
            "compute": {"source": "training logs and SLURM", "params_m": 9.1, "gpu": "one NVIDIA H100 (80 GB)",
                        "hours_per_run": "about 2 to 3 (C25 2:31, C26 2:21, C27 2:58, c29 2:01)",
                        "peak_gb": 37.9},
            "orientation_spin_match": {"source": "DATASET.md", "shared": 35, "independent": 0.3, "gallery": 394},
            "phase_b_patch_params_m": 11.9, "phase_b_pointnet_params_m_per_modality": 1.2,
            "phase_c_runs": len([p for p in (CLIP / "phase_c").glob("c*") if p.is_dir()
                                 and p.name != "c28_combined_seed3"]),
        },
        "timeline": [
            ["22 Oct", "repository started: modules shared with GalaxyCLIP-2D, 2D-3D pairing tables"],
            ["27 Oct", "audit of the existing snapshot-99 copy: 109 files missing"],
            ["2 Nov", "fresh complete copy of snapshot 99 downloaded (2.92 TB)"],
            ["9 Nov", "point-cloud dataset v3 built (5,254 galaxies, 4 modalities); literature search"],
            ["13 Nov", "alignment plan approved; infrastructure, baselines, orientation shortcut; Phase B runs"],
            ["15 Nov", "Phase C: 28 runs in four waves (one change at a time, then combinations); final setup chosen"],
            ["16 Nov", "Phase D: third seed, test set scored once, linear probes"],
            ["17 Nov", "thesis figures and tables; code committed and pushed; dark-matter frame control"],
            ["18 Nov", "dark-matter frame result; matched-subset control"],
        ],
    }
    reading = Path(__file__).with_name("c30_reading.json")        # the written reading of the matched control
    if n["c30"] and reading.exists():
        n.update(load(reading))
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1) + "\n")
    print(f"wrote {OUT / 'numbers.json'}: {len(tables)} tables, c30 {'present' if n['c30'] else 'pending'}")
    print("dataset:", {k: n['dataset'][k] for k in ('selected', 'selected_centrals', 'halos', 'usable_r5')},
          n["dataset"]["splits"])
    print("rvmax subset:", n["rvmax_subset"])
    print("baselines test:", {k: [round(v['r1_10dir'], 1), round(v['r1_12dir'], 1)] for k, v in n['baselines_test'].items()})
    print("final config:", n["final_config"])
    print("final train:", n["final_train"])


if __name__ == "__main__":
    main()
