"""Per-domain memory queue of extra InfoNCE negatives (MoCo / XBM style).

The logic is GalaxyCLIP-2D's queue (commit 8b0a377, GalaxyCLIPMultiDomain.enqueue /
set_queue_size / queue_negatives), moved out of the model class into a standalone
object so any encoder can use it. Behaviour is unchanged:

* read the negatives BEFORE enqueuing the current batch, so a batch never sees its
  own positives as queued negatives;
* under DDP, enqueue the all-gathered embeddings so every rank holds the same queue;
* the queue is not checkpointed -- it refills within K / global_batch steps after a
  resume;
* grow it gradually (queue_size_at_epoch in training.py): a single large jump
  collapsed retrieval in the 2D runs.
"""
from __future__ import annotations

import torch


class EmbeddingQueue:
    def __init__(self, domains: list[str], queue_size: int = 0):
        self.domains = list(domains)
        self.queue_size = int(queue_size)
        self._queue = None          # (n_domains, K, embed_dim), lazily created
        self._queue_ptr = 0
        self._queue_filled = 0

    def enqueue(self, embeddings: dict[str, torch.Tensor]) -> None:
        """Push a batch of embeddings into the per-domain circular queue (detached)."""
        if self.queue_size <= 0 or not embeddings:
            return
        ref = next(iter(embeddings.values()))
        if self._queue is None:
            self._queue = torch.zeros(len(self.domains), self.queue_size,
                                      ref.size(1), device=ref.device,
                                      dtype=ref.dtype)
        b = min(ref.size(0), self.queue_size)
        idx = (torch.arange(b, device=ref.device) + self._queue_ptr) % self.queue_size
        for i, d in enumerate(self.domains):
            if d in embeddings:
                self._queue[i].index_copy_(
                    0, idx, embeddings[d][-b:].detach().to(self._queue.dtype))
        self._queue_ptr = int((self._queue_ptr + b) % self.queue_size)
        self._queue_filled = int(min(self._queue_filled + b, self.queue_size))

    @torch.no_grad()
    def set_queue_size(self, new_size: int) -> bool:
        """Resize in place, keeping the entries already held. True if the size changed."""
        new_size = int(new_size)
        if new_size == self.queue_size:
            return False
        if self._queue is None:          # not allocated yet -> just record it
            self.queue_size = new_size
            return True
        keep = min(self._queue_filled, new_size)
        old = self._queue
        self._queue = torch.zeros(len(self.domains), new_size, old.size(2),
                                  device=old.device, dtype=old.dtype)
        if keep:
            self._queue[:, :keep] = old[:, :keep]
        self.queue_size = new_size
        self._queue_filled = keep
        self._queue_ptr = keep % new_size if new_size else 0
        return True

    def negatives(self) -> dict[str, torch.Tensor]:
        """{domain: (n_valid, D)} extra negatives; empty when off / not yet filled."""
        if self.queue_size <= 0 or self._queue is None or self._queue_filled == 0:
            return {}
        n = self._queue_filled
        return {d: self._queue[i, :n] for i, d in enumerate(self.domains)}
