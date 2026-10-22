"""Copied verbatim from GalaxyCLIP-2D (commit 8b0a377), galaxyclip/utils.py.

Utilities: checkpointing, signal-based requeue, DDP helpers, AutoFlusher.

The AutoFlusher pattern follows ZHAW's cluster docs (basics_4_batch_py); the
signal-based requeue follows basics_resubmit/resubmit_pytorch.submit.
"""
from __future__ import annotations

import glob
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch


# ──────────────────────────────────────────────────────────────────────
# AutoFlusher (ZHAW cluster_user/basics_4_batch_py/AutoFlusher.py)
# ──────────────────────────────────────────────────────────────────────
class AutoFlusher:
    """Wrap stdout so every print is flushed immediately into the SLURM .out file.

    Forwards every other attribute (isatty, fileno, encoding, etc.) to the
    underlying stream so callers like torch._dynamo that probe stdout don't
    crash with AttributeError.
    """

    def __init__(self, output_stream: TextIO):
        self.output_stream = output_stream

    def write(self, x: str) -> None:
        self.output_stream.write(x)
        self.output_stream.flush()

    def writelines(self, xs: Iterable[str]) -> None:
        self.output_stream.writelines(xs)
        self.output_stream.flush()

    def flush(self) -> None:
        self.output_stream.flush()

    def __getattr__(self, name):
        # Anything not defined above is forwarded to the wrapped stream.
        # This catches isatty, fileno, encoding, errors, mode, name, closed, etc.
        return getattr(self.output_stream, name)

def install_autoflusher():
    """Idempotent: route stdout/stderr through AutoFlusher."""
    if not isinstance(sys.stdout, AutoFlusher):
        sys.stdout = AutoFlusher(sys.stdout)
    if not isinstance(sys.stderr, AutoFlusher):
        sys.stderr = AutoFlusher(sys.stderr)


# ──────────────────────────────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────────────────────────────
def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    if is_distributed():
        return int(os.environ.get("RANK", 0))
    return 0


def get_world_size() -> int:
    if is_distributed():
        return int(os.environ.get("WORLD_SIZE", 1))
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize torch.distributed. Returns (rank, local_rank, world_size).

    Expects torchrun-style env vars: RANK, LOCAL_RANK, WORLD_SIZE,
    MASTER_ADDR, MASTER_PORT.
    """
    import torch.distributed as dist

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank,
                                world_size=world_size)
        torch.cuda.set_device(local_rank)
    elif world_size == 1:
        torch.cuda.set_device(0)

    return rank, local_rank, world_size


def cleanup_distributed():
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor by mean across the process group."""
    if get_world_size() == 1:
        return tensor
    import torch.distributed as dist
    out = tensor.clone().detach()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= get_world_size()
    return out


def gather_embeddings(z: torch.Tensor) -> torch.Tensor:
    """All-gather embeddings across DDP ranks into one (world*B, D) tensor.

    The local shard keeps its gradient path while the gathered copies of the
    other ranks are detached; combined with DDP's gradient all-reduce this
    reproduces full-batch InfoNCE gradients (the OpenCLIP / SimCLR gather
    pattern). This is what makes scaling GPUs actually enlarge the contrastive
    negative pool — without it, each rank only ever sees its local batch as
    negatives. No-op (returns ``z`` unchanged) when not distributed.
    """
    if not is_distributed() or get_world_size() == 1:
        return z
    import torch.distributed as dist
    if not dist.is_initialized():
        return z
    world = get_world_size()
    rank = get_rank()
    gathered = [torch.zeros_like(z) for _ in range(world)]
    dist.all_gather(gathered, z.contiguous())
    gathered[rank] = z  # restore grad path for the local shard
    return torch.cat(gathered, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Signal-based requeue (SIGUSR1 → save & exit; bash trap requeues)
# ──────────────────────────────────────────────────────────────────────
class RequeueState:
    """Tiny module-level flag set by the SIGUSR1 handler.

    Training loops should poll `state.should_stop` after each batch/epoch
    and call the user-provided save_fn + exit cleanly when it flips.
    """
    should_stop: bool = False
    received_at: float | None = None


_state = RequeueState()


def _sigusr1_handler(signum, frame):
    _state.should_stop = True
    _state.received_at = time.time()
    if is_main_process():
        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] "
              f"Received SIGUSR1 — will checkpoint and exit at next epoch boundary.",
              flush=True)


def install_requeue_handler():
    """Install a SIGUSR1 handler so training jobs can checkpoint before timeout.

    Pair with `#SBATCH --signal=B:SIGUSR1@<seconds>` in the submit file. The
    bash trap in the submit script handles `scontrol requeue` after the
    Python process exits.
    """
    signal.signal(signal.SIGUSR1, _sigusr1_handler)


def should_stop() -> bool:
    return _state.should_stop


# ──────────────────────────────────────────────────────────────────────
# Checkpoints
# ──────────────────────────────────────────────────────────────────────
def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    scaler_state: dict | None = None,
    epoch: int = 0,
    global_step: int = 0,
    history: dict | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Atomic checkpoint write: save to a tmp file then rename.

    Caller is responsible for unwrapping DDP (use `model.module.state_dict()`
    if the model is wrapped in DistributedDataParallel).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "global_step": global_step,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state_dict"] = scheduler_state
    if scaler_state is not None:
        payload["scaler_state_dict"] = scaler_state
    if history is not None:
        payload["history"] = history
    if config is not None:
        payload["config"] = config
    if extra is not None:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location="cpu",
    weights_only: bool = False,
) -> dict:
    return torch.load(str(path), map_location=map_location,
                      weights_only=weights_only)


def find_latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Return the most recently modified checkpoint matching `pattern`, or None."""
    files = glob.glob(str(Path(directory) / pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])


# ──────────────────────────────────────────────────────────────────────
# wandb resume helper
# ──────────────────────────────────────────────────────────────────────
def stable_wandb_id(*parts: str) -> str:
    """Deterministic wandb run id so SLURM requeues continue the same run.

    Combines the parts (e.g. job name + domain pair + seed) into a short
    hex digest — wandb accepts arbitrary strings up to 64 chars.
    """
    import hashlib
    h = hashlib.sha1("/".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]
