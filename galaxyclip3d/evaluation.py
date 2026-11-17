"""Scoring finished runs (Phase D): reload a run, embed a split once, and compute retrieval scores from the embeddings.

`retrieval_scores` is the trainer's validation metric (train_clip.evaluate) computed from embeddings held in memory, so
one pass over the data can serve several models, and the same embeddings feed the linear probes (probes.py).
Embeddings are computed in float32 (training evaluated in bf16 on the GPU, so numbers can differ by a few tenths of a
point from the training log).

    cfg, model, ck = load_run("outputs/clip/phase_c/c21_combined", data=V3)
    out = embed({"c21": model}, run_dataset(cfg, "test"))
    res = retrieval_scores(out["c21"], out["mask"], out["mstar"])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import PointCloudDataset
from .multimodal import MODALITIES, masked_recall
from .train_clip import TrainConfig, build_model
from .utils import load_checkpoint

ITEM_KEYS = ("mask", "subhalo", "snapshot", "mstar")      # per-galaxy values kept next to the embeddings


def load_run(run_dir: str | Path, data: str | Path | None = None, overrides=(), checkpoint: str = "best.pt"):
    """(config, model in eval mode with the checkpoint's weights, checkpoint dict) of a finished run. `data` points the
    run at the dataset's permanent folder (runs record the job's local copy)."""
    run_dir = Path(run_dir)
    sets = ([f"data={data}"] if data is not None else []) + list(overrides)
    cfg = TrainConfig.from_file(str(run_dir / "config.json"), sets)
    model = build_model(cfg)
    ck = load_checkpoint(run_dir / checkpoint)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return cfg, model.eval(), ck


def untrained_model(cfg: TrainConfig) -> torch.nn.Module:
    """The run's architecture with its initial random weights (seeded by the run's seed), never trained."""
    torch.manual_seed(cfg.seed)
    return build_model(cfg).eval()


def run_dataset(cfg: TrainConfig, split: str, rotation: str | None = None, draw_seed: int | None = None,
                standardized: bool = True) -> PointCloudDataset:
    """The run's data settings (as train_clip.build_datasets) on any split; val and test use their frozen clouds."""
    return PointCloudDataset(cfg.data, split, radius=cfg.radius, snapshots=cfg.snapshots,
                             gas_required=cfg.gas_required, rotation=rotation or cfg.rotation, mirror=cfg.mirror,
                             standardized=standardized, budget=cfg.points or None, common_radii=cfg.common_radii,
                             draw_seed=draw_seed)


def spread(n: int, limit: int) -> list[int]:
    """`limit` indices spread evenly over range(n). Galaxies are stored in subhalo order, which groups them by host
    halo, so the first few hundred would come from a handful of hosts."""
    return sorted({int(i) for i in np.linspace(0, n - 1, min(limit, n))})


@torch.no_grad()
def embed(models: dict, dataset, device: str = "cpu", batch_size: int = 64, workers: int = 4, limit: int | None = None,
          per_cloud=None) -> dict:
    """Embeddings of every galaxy in `dataset` under each model, in one pass over the data.

    Returns {name: {modality: float32 array (n, D)}} for each model, the per-galaxy ITEM_KEYS the items carry, and,
    with `per_cloud` (a function of one cloud array (N, F) -> 1-d array), {"per_cloud": {modality: (n, k)}}.
    `limit` keeps that many galaxies spread over the split (quick checks)."""
    dev = torch.device(device)
    for model in models.values():
        model.to(dev).eval()
    data = dataset if limit is None else torch.utils.data.Subset(dataset, spread(len(dataset), limit))
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=False, num_workers=workers)
    embs = {name: {m: [] for m in MODALITIES} for name in models}
    extra: dict[str, list] = {}
    stats = {m: [] for m in MODALITIES}
    for batch in loader:
        clouds = {m: batch[m].to(dev) for m in MODALITIES}
        for name, model in models.items():
            out = model(clouds)
            for m in MODALITIES:
                embs[name][m].append(out[m].float().cpu())
        for k in ITEM_KEYS:
            if k in batch:
                extra.setdefault(k, []).append(torch.as_tensor(batch[k]))
        if per_cloud is not None:
            for m in MODALITIES:
                stats[m].extend(per_cloud(c) for c in batch[m].numpy())
    res = {name: {m: torch.cat(v).numpy() for m, v in e.items()} for name, e in embs.items()}
    res.update({k: torch.cat(v).numpy() for k, v in extra.items()})
    if per_cloud is not None:
        res["per_cloud"] = {m: np.stack(v) for m, v in stats.items()}
    return res


def retrieval_scores(emb: dict, mask, mstar=None, ks=(1, 5, 10), mass_bin_dex: float = 0.5) -> dict[str, float]:
    """Full-gallery recall per modality pair (masked_recall) plus recall among galaxies of similar stellar mass, as
    train_clip.evaluate computes them during training."""
    e = {m: torch.as_tensor(np.asarray(v)) for m, v in emb.items() if m in MODALITIES}
    mk = torch.as_tensor(np.asarray(mask))
    res = masked_recall(e, mk, tuple(ks))
    if mstar is not None and mass_bin_dex > 0:
        logm = torch.log10(torch.as_tensor(np.asarray(mstar)).double().clamp_min(1.0))
        bins = torch.floor(logm / mass_bin_dex).long()
        res.update({f"massbin_{k}": v for k, v in masked_recall(e, mk, (1,), groups=bins).items()
                    if k.startswith(("r1/", "chance/mean"))})
    res["galaxies"] = int(len(mk))
    return res
