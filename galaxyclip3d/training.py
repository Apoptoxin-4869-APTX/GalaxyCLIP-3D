"""Training utilities: adaptive gradient clipping (ZClip), full-gallery recall, progress printing,
memory-queue size schedule, plus a weight EMA helper.

Copied verbatim from GalaxyCLIP-2D (commit 8b0a377), train_encoder_multi.py.
Keep changes deliberate: these functions produced the 2D results.
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.distributed as dist

from .utils import get_world_size


class ZClipState:
    """Adaptive gradient clipping on the z-score of the gradient norm.

    A fixed `grad_clip` assumes the gradient-norm distribution is stationary.
    It is not: job 47321's norm fell 18.6 -> 5.4 over 50 epochs, and 47326's
    4.3 -> 0.6. So one threshold is wrong at both ends -- grad_clip=10 was a
    no-op by the time 47211 spiked (train 0.096 -> 4.23 in one epoch), and
    grad_clip=1.0 fires on 125/125 steps and acts as a normalizer.

    Instead track an EMA of the norm's mean and variance and clip only when
    `norm > mean + z*std`, so the threshold follows the decay and stays a fuse.

    Two details that matter:
      * On a spike the EMA is updated with the CLIPPED value, not the raw norm.
        Feeding the outlier back in would ratchet the threshold up and disarm
        the clipper exactly when a run of spikes makes it most necessary.
      * The criterion is inactive until `warmup_steps` norms have been seen,
        otherwise the first observation defines the statistics. During seeding
        no clipping is applied -- gradients are only measured.

    Pure float arithmetic, so it is unit-testable without a GPU.
    """

    def __init__(self, z: float = 2.5, alpha: float = 0.97,
                 warmup_steps: int = 100):
        self.z, self.alpha, self.warmup_steps = z, alpha, warmup_steps
        self.mean, self.var, self.n = 0.0, 0.0, 0
        self.n_spikes = 0

    @property
    def ready(self) -> bool:
        return self.n >= self.warmup_steps

    def threshold(self) -> float:
        """Current clip threshold; inf while still seeding."""
        if not self.ready:
            return float("inf")
        return self.mean + self.z * math.sqrt(max(self.var, 0.0))

    def update(self, norm: float, threshold: float) -> bool:
        """Fold `norm` into the statistics. Returns True if it was a spike."""
        if norm != norm or norm == float("inf"):      # NaN / inf
            return False
        spike = self.ready and norm > threshold
        obs = threshold if spike else norm
        if spike:
            self.n_spikes += 1
        delta = obs - self.mean
        if self.n == 0:
            self.mean, self.var = obs, 0.0
        else:
            self.mean += (1.0 - self.alpha) * delta
            self.var = self.alpha * (self.var + (1.0 - self.alpha) * delta * delta)
        self.n += 1
        return spike


def _all_gather_cat(z: torch.Tensor) -> torch.Tensor:
    """All-gather equal-sized (n, D) shards and concatenate in rank order.

    The val DistributedSampler uses drop_last=False, which pads so every rank
    holds the same count -- so shapes match. We assert that rather than assume
    it: an uneven gather would silently misalign domain a's row i from domain
    b's row i and quietly corrupt the metric.
    """
    ws = get_world_size()
    n = torch.tensor([z.shape[0]], device=z.device)
    sizes = [torch.zeros_like(n) for _ in range(ws)]
    dist.all_gather(sizes, n)
    counts = [int(s.item()) for s in sizes]
    if len(set(counts)) != 1:
        raise RuntimeError(f"uneven val shards across ranks: {counts}")
    out = [torch.empty_like(z) for _ in range(ws)]
    dist.all_gather(out, z.contiguous())
    return torch.cat(out, dim=0)


def full_gallery_recall(embeddings: dict[str, torch.Tensor],
                        ks: tuple[int, ...],
                        chunk: int = 1024) -> dict[str, float]:
    """Recall@k over the FULL val gallery, meaned over all 42 ordered pairs.

    `cross_domain_retrieval_accuracy` scores each query against its own batch
    only (gallery = batch_size_per_gpu). This scores every val galaxy against
    every other one (gallery = the whole split), so the two numbers are NOT
    comparable and are logged separately.

    Rank is the count of gallery entries scoring strictly above the true
    partner, which needs no top-k sort and does not let ties inflate the score.
    Queries are chunked so the similarity block stays (chunk, N).

    CAVEAT for interpretation: the group-aware split keeps all 4 rotations of a
    galaxy on the same side, so the gallery holds ~3 near-duplicate distractors
    per query. Retrieving a rotation of the correct galaxy counts as a miss.
    """
    doms = list(embeddings.keys())
    n = embeddings[doms[0]].shape[0]
    hits = {k: 0.0 for k in ks}
    per_pair: dict[str, float] = {}
    n_dir = 0
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            za, zb = embeddings[doms[i]], embeddings[doms[j]]
            for q, g in ((za, zb), (zb, za)):
                ranks = torch.empty(n, dtype=torch.long, device=q.device)
                for s in range(0, n, chunk):
                    e = min(s + chunk, n)
                    sim = q[s:e] @ g.t()                       # (C, N)
                    idx = torch.arange(s, e, device=q.device).unsqueeze(1)
                    true = sim.gather(1, idx)                  # (C, 1)
                    ranks[s:e] = (sim > true).sum(dim=1)
                qd = doms[i] if q is za else doms[j]
                gd = doms[j] if q is za else doms[i]
                for k in ks:
                    v = (ranks < k).float().mean().item()
                    hits[k] += v
                    if k == ks[0]:
                        # Per-pair breakdown at the smallest k (normally R@1). Flat
                        # scalar keys so the history/wandb loops that iterate this
                        # dict keep working unchanged.
                        per_pair[f"pair{k}_{qd}->{gd}"] = v
                n_dir += 1
    out = {f"r@{k}": hits[k] / max(n_dir, 1) for k in ks}
    out.update(per_pair)
    return out


def _hms(seconds: float) -> str:
    """Compact h:mm:ss / m:ss for progress lines."""
    s = int(max(seconds, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class EpochProgress:
    """Rank-0 in-epoch progress reporter.

    Interactive (stdout is a TTY): redraws one bar per second with \r.
    Batch job (no TTY): prints a fresh line every `pct` percent, so an 8-hour
    SLURM log gains ~20 lines per epoch instead of thousands of \r fragments.
    """

    def __init__(self, total, epoch, num_epochs, pct=5, enabled=True):
        self.total = max(int(total), 1)
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.pct = max(int(pct), 0)
        self.enabled = bool(enabled) and self.pct > 0
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.start = time.time()
        self.next_mark = float(self.pct)
        self._last_draw = 0.0

    def update(self, batch_idx, loss=None):
        if not self.enabled:
            return
        done = batch_idx + 1
        last = done >= self.total
        frac = done / self.total
        pct_done = frac * 100.0
        now = time.time()
        if self.tty:
            if not last and now - self._last_draw < 1.0:
                return
            self._last_draw = now
        else:
            if not last and pct_done < self.next_mark:
                return
            while self.next_mark <= pct_done:
                self.next_mark += self.pct
        elapsed = now - self.start
        rate = done / max(elapsed, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        nblocks = 24
        filled = int(nblocks * frac)
        bar = "#" * filled + "-" * (nblocks - filled)
        msg = (f"    ep {self.epoch + 1:3d}/{self.num_epochs} "
               f"[{bar}] {pct_done:5.1f}%  {done}/{self.total}  "
               f"{rate:.2f} it/s  elapsed {_hms(elapsed)}  eta {_hms(eta)}")
        if loss is not None:
            msg += f"  loss={loss:.4f}"
        print(msg, end=("\r" if self.tty else "\n"), flush=True)

    def close(self):
        if self.enabled and self.tty:
            print(flush=True)


def queue_size_at_epoch(cfg, epoch: int) -> int:
    """Effective queue size for a 0-indexed epoch (0 = queue entirely off).

    Three shapes, in order of precedence:
      ramp      queue_size + (elapsed // ramp_every) * ramp_step, capped by
                queue_ramp_max when that is > 0
      two-stage queue_size, then queue_size_2 from queue_start_epoch_2 on
      fixed     queue_size
    """
    if epoch < cfg.queue_start_epoch:
        return 0
    if cfg.queue_ramp_every > 0 and cfg.queue_ramp_step > 0:
        n = (epoch - cfg.queue_start_epoch) // cfg.queue_ramp_every
        k = cfg.queue_size + n * cfg.queue_ramp_step
        if cfg.queue_ramp_max > 0:
            k = min(k, cfg.queue_ramp_max)
        return max(int(k), 0)
    if (cfg.queue_size_2 > 0 and cfg.queue_start_epoch_2 > 0
            and epoch >= cfg.queue_start_epoch_2):
        return int(cfg.queue_size_2)
    return int(cfg.queue_size)


class ModelEMA:
    """Exponential moving average of a model's parameters (new in GalaxyCLIP-3D).

    Same update as the EMA option added to GalaxyCLIP-2D's trainer: after every
    optimizer step, ema = d * ema + (1 - d) * live, via one fused foreach lerp.
    `ema_model` must be a separate instance of the same architecture (not wrapped
    in DDP); evaluate it like any model. d averages over ~1/(1-d) steps.
    """

    def __init__(self, live_model: torch.nn.Module, ema_model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.model = ema_model
        self.model.load_state_dict(live_model.state_dict())
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._ema = list(self.model.parameters())
        self._live = [p.detach() for p in live_model.parameters()]
        if len(self._ema) != len(self._live) or any(a.shape != b.shape for a, b in zip(self._ema, self._live)):
            raise RuntimeError("EMA and live model parameters do not line up")
        self._live_model = live_model

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self._ema, self._live, 1.0 - self.decay)

    @torch.no_grad()
    def sync_buffers(self) -> None:
        for b_e, b in zip(self.model.buffers(), self._live_model.buffers()):
            b_e.copy_(b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)
