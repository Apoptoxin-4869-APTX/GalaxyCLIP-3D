"""Train the multi-modal contrastive model on the point-cloud dataset (one GPU).

    PYTHONPATH=pylibs:. python -s -m galaxyclip3d.train_clip --config configs/clip_pointnet.json [--set key=value ...]

Each training item is a galaxy's four clouds, drawn fresh with an independent random rotation per modality; the loss
is the 2D project's all-pairs symmetric InfoNCE, per pair over the galaxies that have both modalities
(multimodal.py). Validation uses the frozen clouds with fixed per-modality rotations and full-gallery recall over all
validation galaxies. Before the first epoch the untrained model is evaluated too (epoch 0).

Writes into out_dir: latest.pt (everything needed to resume; a SIGUSR1 from SLURM saves it and exits with code 3),
best.pt (weights at the best validation R@1, mean over pairs without gas-HI), history.json, config.json. Rerunning
with the same out_dir resumes from latest.pt.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .dataset import MassBalancedSampler, PointCloudDataset
from .models import MultiModalCLIP
from .multimodal import MODALITIES, masked_all_pairs_loss, masked_recall, pairs, shuffle_partners
from .pointclouds.sampling import FEATURES, INPUTS, input_columns
from .training import ModelEMA, ZClipState
from .utils import install_requeue_handler, load_checkpoint, save_checkpoint, should_stop, stable_wandb_id


@dataclass
class TrainConfig:
    data: str = "/cluster/projects/ska/illustris/tng50-1.pointclouds/v3"
    out_dir: str = "outputs/clip/run"
    run_name: str = "clip"
    snapshots: list | None = None             # None = every snapshot in the dataset
    radius: float = 5.0                       # sphere in stellar half-mass radii
    rotation: str = "independent"            # independent | shared | none (controls)
    mirror: bool = True
    gas_required: bool = False
    points: int = 0                           # points per cloud; 0 = the dataset's budget (2,048)
    common_radii: list | None = None          # keep only galaxies usable (modalities on) at every one of these spheres
    inputs: str = "all"                       # all | no_velocities | no_scalars (columns the encoders see)
    point_invariants: bool = False            # also feed each point's distance, speed, radial and tangential velocity
    # model
    encoder: str = "patch"                   # patch | pointnet
    width: int = 384
    depth: int = 6
    heads: int = 6
    patches: int = 128
    patch_size: int = 32
    patch_centres: str = "fps"                # fps | radial (centres picked in radially equalised positions)
    pointnet_widths: list = field(default_factory=lambda: [64, 128, 256, 512])
    shared_encoder: bool = False              # one PointNet for all modalities (own first layer and head each)
    dropout: float = 0.0
    encoder_dim: int = 512
    embed_dim: int = 512
    temperature: float = 0.07
    learn_temperature: bool = False
    loss_pairs: str = "all"                   # all | stars_anchor (train only the 3 stars-X pairs; val keeps all 12)
    # optimisation
    batch_size: int = 256
    epochs: int = 500
    lr: float = 5e-4
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    zclip: bool = True
    zclip_z: float = 2.5
    ema_decay: float = 0.0                    # 0 = off; else evaluate and save the EMA weights
    amp: str = "bf16"                         # bf16 | fp16 | none
    sampler: str = "uniform"                  # uniform | mass_balanced
    workers: int = 16
    seed: int = 4869
    control: str = "none"                     # none | shuffle_pairs (leak test: must stay at chance)
    # evaluation and logging
    eval_every: int = 5
    ks: list = field(default_factory=lambda: [1, 5, 10])
    mass_bin_dex: float = 0.5                 # bin width for mass-matched retrieval (0 = off)
    max_steps_per_epoch: int = 0              # > 0 only for smoke tests
    wandb: bool = False
    wandb_project: str = "galaxyclip3d"

    def __post_init__(self):
        for k, ok in (("inputs", INPUTS), ("loss_pairs", ("all", "stars_anchor")), ("patch_centres", ("fps", "radial"))):
            if getattr(self, k) not in ok:
                raise SystemExit(f"{k} must be one of {ok}, got {getattr(self, k)!r}")
        if self.shared_encoder and self.encoder != "pointnet":
            raise SystemExit("shared_encoder needs encoder=pointnet")

    @classmethod
    def from_file(cls, path: str | None, overrides: list[str]) -> "TrainConfig":
        d = json.loads(Path(path).read_text()) if path else {}
        known = {f.name: f for f in dataclasses.fields(cls)}
        for item in overrides:
            k, _, v = item.partition("=")
            if k not in known:
                raise SystemExit(f"unknown setting {k!r}")
            try:
                d[k] = json.loads(v)
            except json.JSONDecodeError:
                d[k] = v
        unknown = set(d) - set(known)
        if unknown:
            raise SystemExit(f"unknown settings {sorted(unknown)}")
        return cls(**d)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_model(cfg: TrainConfig) -> MultiModalCLIP:
    dims = {m: len(FEATURES[m]) for m in MODALITIES}
    columns = None if cfg.inputs == "all" else {m: input_columns(m, cfg.inputs) for m in MODALITIES}
    return MultiModalCLIP(dims, cfg.encoder, cfg.embed_dim, cfg.encoder_dim, cfg.temperature, cfg.learn_temperature,
                          columns=columns, shared_encoder=cfg.shared_encoder, point_invariants=cfg.point_invariants,
                          width=cfg.width, depth=cfg.depth, heads=cfg.heads, patches=cfg.patches,
                          patch_size=cfg.patch_size, pointnet_widths=tuple(cfg.pointnet_widths), dropout=cfg.dropout,
                          patch_centres=cfg.patch_centres)


def build_datasets(cfg: TrainConfig):
    kw = dict(radius=cfg.radius, snapshots=cfg.snapshots, gas_required=cfg.gas_required, rotation=cfg.rotation,
              mirror=cfg.mirror, budget=cfg.points or None, common_radii=cfg.common_radii)
    return PointCloudDataset(cfg.data, "train", **kw), PointCloudDataset(cfg.data, "val", **kw)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "cls" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def to_device(batch: dict, device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    clouds = {m: batch[m].to(device, non_blocking=True) for m in MODALITIES}
    return clouds, batch["mask"].to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, device, cfg: TrainConfig, amp_dtype) -> dict[str, float]:
    model.eval()
    embs = {m: [] for m in MODALITIES}
    masks, mstar = [], []
    for batch in loader:
        clouds, mask = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(clouds)
        for m in MODALITIES:
            embs[m].append(out[m].float())
        masks.append(mask)
        mstar.append(batch["mstar"])
    emb = {m: torch.cat(v) for m, v in embs.items()}
    mask = torch.cat(masks)
    res = masked_recall(emb, mask, tuple(cfg.ks))
    if cfg.mass_bin_dex > 0:
        logm = torch.log10(torch.cat(mstar).double().clamp_min(1.0))
        bins = torch.floor(logm / cfg.mass_bin_dex).long().to(device)
        res.update({f"massbin_{k}": v for k, v in masked_recall(emb, mask, (1,), groups=bins).items()
                    if k.startswith(("r1/", "chance/mean"))})
    res["galaxies"] = int(len(mask))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = TrainConfig.from_file(args.config, args.set)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1) + "\n")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed % 2**32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds, val_ds = build_datasets(cfg)
    loader_kw = dict(num_workers=cfg.workers, pin_memory=device.type == "cuda",
                     persistent_workers=cfg.workers > 0)
    if cfg.sampler == "mass_balanced":
        sampler = MassBalancedSampler(train_ds, seed=cfg.seed)
        train_loader = torch.utils.data.DataLoader(train_ds, cfg.batch_size, sampler=sampler, drop_last=True,
                                                   **loader_kw)
    else:
        sampler = None
        train_loader = torch.utils.data.DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                                                   generator=torch.Generator().manual_seed(cfg.seed), **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, cfg.batch_size, shuffle=False, **loader_kw)
    steps_per_epoch = len(train_loader) if not cfg.max_steps_per_epoch else min(len(train_loader),
                                                                                cfg.max_steps_per_epoch)

    model = build_model(cfg).to(device)
    ema_model = build_model(cfg).to(device) if cfg.ema_decay > 0 else None
    ema = ModelEMA(model, ema_model, cfg.ema_decay) if ema_model is not None else None
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=tuple(cfg.betas))
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, cfg.warmup_epochs * steps_per_epoch, total, cfg.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    zclip = ZClipState(z=cfg.zclip_z) if cfg.zclip else None
    n_params = sum(p.numel() for p in model.parameters())

    history: dict = {"epoch": [], "train_loss": [], "pair_loss": [], "lr": [], "temperature": [], "epoch_s": [],
                     "data_wait_s": [], "val": []}
    start_epoch, step, best = 0, 0, -1.0
    latest = out / "latest.pt"
    if latest.exists():
        ck = load_checkpoint(latest)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        if ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        if ema is not None and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        if zclip is not None and ck.get("zclip"):
            zclip.__dict__.update(ck["zclip"])
        start_epoch, step, history, best = ck["epoch"], ck["global_step"], ck["history"], ck.get("best", -1.0)
        log(f"resumed from {latest} at epoch {start_epoch}")

    wb = None
    if cfg.wandb:
        try:
            import wandb
            wb = wandb.init(project=cfg.wandb_project, name=cfg.run_name, id=stable_wandb_id(cfg.run_name, str(out)),
                            resume="allow", config=dataclasses.asdict(cfg), dir=str(out))
        except Exception as e:                                            # noqa: BLE001
            log(f"W&B unavailable ({e}); logging to history.json only")
    install_requeue_handler()
    log(f"{cfg.encoder} encoders, {n_params / 1e6:.1f} M parameters, on {device}; train {len(train_ds):,} galaxies "
        f"({steps_per_epoch} steps of {cfg.batch_size}), val {len(val_ds):,}; rotation {cfg.rotation}"
        f"{' + mirror' if cfg.mirror else ''}; control {cfg.control}" +
        "".join(f"; {k} {getattr(cfg, k)}" for k in ("points", "common_radii", "inputs", "point_invariants",
                                                     "patch_centres", "shared_encoder", "loss_pairs")
                if getattr(cfg, k) != TrainConfig.__dataclass_fields__[k].default))

    def eval_model():
        return evaluate(ema_model if ema_model is not None else model, val_loader, device, cfg, amp_dtype)

    def save_latest(epoch):
        save_checkpoint(latest, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                        scheduler_state=sched.state_dict(), scaler_state=scaler.state_dict(), epoch=epoch,
                        global_step=step, history=history, config=dataclasses.asdict(cfg),
                        extra={"best": best, "ema": ema.state_dict() if ema is not None else None,
                               "zclip": dict(zclip.__dict__) if zclip is not None else None})

    def report(epoch, res):
        history["val"].append({"epoch": epoch, **res})
        (out / "history.json").write_text(json.dumps(history) + "\n")
        log(f"val epoch {epoch}: R@1 {res['r1/mean']:.1%} (without gas-HI {res['r1/mean_no_gas_hi']:.1%}), "
            f"R@5 {res['r5/mean']:.1%}, chance {res['chance/mean']:.2%}; mass-matched R@1 "
            f"{res.get('massbin_r1/mean_no_gas_hi', float('nan')):.1%} (chance "
            f"{res.get('massbin_chance/mean', float('nan')):.1%}); " +
            ", ".join(f"{k[3:]} {v:.0%}" for k, v in res.items() if k.startswith("r1/") and "->" in k))
        if wb is not None:
            wb.log({f"val/{k}": v for k, v in res.items()}, step=step)

    if start_epoch == 0 and not history["val"]:
        report(0, eval_model())

    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    train_pairs = pairs(MODALITIES, anchor="stars" if cfg.loss_pairs == "stars_anchor" else None)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0, wait, losses, pair_sum, n = time.time(), 0.0, 0.0, {}, 0
        tw = time.time()
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            wait += time.time() - tw
            clouds, mask = to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                emb = model(clouds)
            if cfg.control == "shuffle_pairs":
                emb, mask = shuffle_partners(emb, mask, gen)
            loss, pair = masked_all_pairs_loss(emb, mask, model.temperature(), pair_list=train_pairs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
            if zclip is not None:
                thr = zclip.threshold()
                if zclip.update(norm, thr):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), thr)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update()
            step += 1
            losses += float(loss.detach())
            for k, v in pair.items():
                pair_sum[k] = pair_sum.get(k, 0.0) + v
            n += 1
            if should_stop():
                save_latest(epoch)
                log(f"stop signal: saved {latest} at epoch {epoch} (step {step}); exiting for requeue")
                sys.exit(3)
            tw = time.time()
        epoch_s = time.time() - t0
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(losses / max(n, 1))
        history["pair_loss"].append({k: v / max(n, 1) for k, v in pair_sum.items()})
        history["lr"].append(sched.get_last_lr()[0])
        history["temperature"].append(float(model.temperature().detach()))
        history["epoch_s"].append(epoch_s)
        history["data_wait_s"].append(wait)
        gpu_gb = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
        history.setdefault("gpu_peak_gb", []).append(gpu_gb)       # peak since the start of this process
        log(f"epoch {epoch + 1}/{cfg.epochs}: loss {losses / max(n, 1):.3f}, lr {sched.get_last_lr()[0]:.2e}, "
            f"temperature {float(model.temperature().detach()):.3f}, {epoch_s:.1f} s ({wait / max(epoch_s, 1e-9):.0%} waiting "
            f"for data)" + (f", GPU memory peak {gpu_gb:.1f} GB" if device.type == "cuda" else "")
            + (f", {zclip.n_spikes} clipped steps" if zclip is not None else ""))
        if wb is not None:
            wb.log({"train/loss": losses / max(n, 1), "train/lr": sched.get_last_lr()[0], "train/epoch_s": epoch_s,
                    "train/temperature": float(model.temperature().detach()), "train/gpu_peak_gb": gpu_gb,
                    **{f"train/{k}": v / max(n, 1) for k, v in pair_sum.items()}}, step=step)
        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
            res = eval_model()
            report(epoch + 1, res)
            if res["r1/mean_no_gas_hi"] > best:
                best = res["r1/mean_no_gas_hi"]
                weights = (ema_model if ema_model is not None else model).state_dict()
                save_checkpoint(out / "best.pt", model_state=weights, epoch=epoch + 1, global_step=step,
                                config=dataclasses.asdict(cfg), extra={"val": res})
        save_latest(epoch + 1)
    (out / "history.json").write_text(json.dumps(history) + "\n")
    log(f"done: best val R@1 without gas-HI {best:.1%}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
