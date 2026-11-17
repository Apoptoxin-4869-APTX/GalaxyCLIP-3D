"""Contrastive loss and retrieval for galaxies whose gas or HI modality can be missing (masked).

The 2D functions in losses.py and training.py assume every domain exists for every sample. Here each modality pair
uses only the galaxies that have both modalities: the pair's loss is the 2D symmetric InfoNCE (losses.info_nce_loss)
on those rows, and in retrieval its gallery is those galaxies. Pairs are averaged with equal weight.

`mask` is a (galaxies, 4) bool tensor in MODALITIES order (stars, dm, gas, hi), as the dataset returns it.
"""
from __future__ import annotations

from itertools import combinations

import torch

from .losses import info_nce_loss

MODALITIES = ("stars", "dm", "gas", "hi")
NEAR_DUPLICATE = {("gas", "hi")}          # drawn from the same gas cells; reported separately, like 2D's hi/hi-21cm


def pairs(modalities=MODALITIES, anchor: str | None = None) -> list[tuple[str, str]]:
    """Unordered modality pairs; with `anchor`, only the pairs that include it (an ImageBind-style anchor)."""
    every = list(combinations(modalities, 2))
    if anchor is None:
        return every
    if anchor not in modalities:
        raise ValueError(f"anchor {anchor!r} is not one of {modalities}")
    return [p for p in every if anchor in p]


def masked_all_pairs_loss(emb: dict[str, torch.Tensor], mask: torch.Tensor, temperature,
                          modalities=MODALITIES, min_rows: int = 2,
                          pair_list: list[tuple[str, str]] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean over modality pairs (all, or `pair_list`) of the symmetric InfoNCE on the galaxies that have both. Pairs
    with fewer than `min_rows` such galaxies in the batch are skipped. Returns (loss, {"a-b": pair loss})."""
    total, n, per_pair = None, 0, {}
    for a, b in pairs(modalities) if pair_list is None else pair_list:
        rows = mask[:, modalities.index(a)] & mask[:, modalities.index(b)]
        if int(rows.sum()) < min_rows:
            continue
        loss = info_nce_loss(emb[a][rows].float(), emb[b][rows].float(), temperature)
        per_pair[f"{a}-{b}"] = float(loss.detach())
        total = loss if total is None else total + loss
        n += 1
    if total is None:
        raise ValueError("no modality pair has enough galaxies in this batch")
    return total / n, per_pair


def shuffle_partners(emb: dict[str, torch.Tensor], mask: torch.Tensor, generator: torch.Generator | None = None,
                     modalities=MODALITIES) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Control for leaks: pair every galaxy's stars with other galaxies' dark matter, gas and HI (a fresh random
    permutation per modality). Nothing true is left to learn, so held-out retrieval must stay at chance."""
    out, new_mask = dict(emb), mask.clone()
    for k, mod in enumerate(modalities[1:], start=1):
        perm = torch.randperm(len(mask), generator=generator,
                              device=generator.device if generator is not None else mask.device)
        out[mod] = emb[mod][perm]
        new_mask[:, k] = mask[perm, k]
    return out, new_mask


@torch.no_grad()
def masked_recall(emb: dict[str, torch.Tensor], mask: torch.Tensor, ks=(1, 5, 10), groups: torch.Tensor | None = None,
                  modalities=MODALITIES) -> dict[str, float]:
    """Full-gallery retrieval for every ordered modality pair, over the galaxies that have both modalities.

    The rank of the true partner is the number of other gallery entries scoring at least as high: ties count against
    the query, so a collapsed model whose embeddings are all equal scores zero rather than 100%.
    With `groups` (an int per galaxy, e.g. a stellar-mass bin), each query only searches galaxies of its own group.
    Keys: r{k}/{q}->{g}, medrank/{q}->{g}, gallery/{q}->{g}, chance/{q}->{g}, and r{k}/mean, r{k}/mean_no_gas_hi
    (mean over directions, without the near-duplicate gas-HI pair).
    """
    out: dict[str, float] = {}
    names, plain = [], []
    for a, b in pairs(modalities):
        rows = mask[:, modalities.index(a)] & mask[:, modalities.index(b)]
        idx = rows.nonzero().squeeze(1)
        n = len(idx)
        if n < 2:
            continue
        za, zb = emb[a][idx].float(), emb[b][idx].float()
        same = None if groups is None else groups[idx][:, None] == groups[idx][None, :]
        for q, g, name in ((za, zb, f"{a}->{b}"), (zb, za, f"{b}->{a}")):
            sim = q @ g.t()
            true = sim.diagonal().clone()
            size = torch.full((n,), float(n), device=sim.device)
            if same is not None:
                sim = sim.masked_fill(~same, float("-inf"))
                size = same.sum(1).float()
            rank = (sim >= true[:, None]).sum(1) - 1
            for k in ks:
                out[f"r{k}/{name}"] = float((rank < k).float().mean())
            out[f"medrank/{name}"] = float(rank.float().median()) + 1.0
            out[f"gallery/{name}"] = float(size.mean())
            out[f"chance/{name}"] = float((1.0 / size).mean())
            names.append(name)
            if (a, b) not in NEAR_DUPLICATE:
                plain.append(name)
    for k in ks:
        out[f"r{k}/mean"] = sum(out[f"r{k}/{x}"] for x in names) / max(len(names), 1)
        out[f"r{k}/mean_no_gas_hi"] = sum(out[f"r{k}/{x}"] for x in plain) / max(len(plain), 1)
    out["chance/mean"] = sum(out[f"chance/{x}"] for x in names) / max(len(names), 1)
    return out
