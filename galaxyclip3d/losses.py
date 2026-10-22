"""Contrastive losses (symmetric InfoNCE: pairwise, anchor, all-pairs) and in-batch retrieval accuracy.

Copied verbatim from GalaxyCLIP-2D (commit 8b0a377), galaxyclip/models.py.
Keep changes deliberate: these functions produced the 2D results.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def info_nce_loss(z_a, z_b, temperature=0.07, queue_a=None, queue_b=None):
    """Symmetric InfoNCE (CLIP-style). z_a, z_b are L2-normalized (B, D).

    queue_a / queue_b: optional (K, D) banks of extra *negative* embeddings for
    the respective domain (MoCo/XBM-style memory queue). They are appended to
    the logits as additional columns only — the positive stays the in-batch
    diagonal, so the targets are unchanged and the queue purely enlarges the
    contrastive denominator (B + K candidates instead of B).
    """
    labels = torch.arange(len(z_a), device=z_a.device)
    logits_ab = torch.mm(z_a, z_b.t()) / temperature
    logits_ba = torch.mm(z_b, z_a.t()) / temperature
    # a→b retrieves among b's: extra negatives must come from domain b's queue.
    if queue_b is not None and queue_b.numel():
        logits_ab = torch.cat(
            [logits_ab, torch.mm(z_a, queue_b.to(z_a.dtype).t()) / temperature], dim=1)
    if queue_a is not None and queue_a.numel():
        logits_ba = torch.cat(
            [logits_ba, torch.mm(z_b, queue_a.to(z_b.dtype).t()) / temperature], dim=1)
    return 0.5 * (F.cross_entropy(logits_ab, labels)
                  + F.cross_entropy(logits_ba, labels))


def anchor_info_nce_loss(embeddings: dict[str, torch.Tensor],
                         anchor_domain: str, temperature: float = 0.07):
    """ImageBind-style loss: sum of InfoNCE(anchor, domain_i) for all non-anchor.

    embeddings: {domain: (B, embed_dim)} L2-normalized tensors.
    Returns (total_loss, per_pair_losses_dict).
    """
    z_anchor = embeddings[anchor_domain]
    total = z_anchor.new_zeros(())
    pair_losses = {}
    for domain, z in embeddings.items():
        if domain == anchor_domain:
            continue
        loss_i = info_nce_loss(z_anchor, z, temperature)
        pair_losses[f"{anchor_domain}↔{domain}"] = loss_i.item()
        total = total + loss_i
    n = max(len(pair_losses), 1)
    return total / n, pair_losses


def all_pairs_info_nce_loss(embeddings: dict[str, torch.Tensor],
                            temperature: float = 0.07,
                            weights: dict[str, float] | None = None,
                            log_vars: dict[str, torch.Tensor] | None = None,
                            queues: dict[str, torch.Tensor] | None = None,
                            tensor_weights: dict[str, torch.Tensor] | None = None):
    """AstroM3-style loss: mean of symmetric InfoNCE over ALL domain pairs.

    Unlike the anchor loss (which aligns N-1 domains to a single anchor), this
    aligns every domain to every other — D*(D-1)/2 pairwise terms for D domains
    (3 terms for 3 domains). All encoders therefore receive a contrastive
    signal from every other domain.

    embeddings: {domain: (B, embed_dim)} L2-normalized tensors. When training
        distributed, B should already be the *gathered* (world*local) batch so
        the negative pool spans all GPUs (see utils.gather_embeddings).

    Per-pair weighting (two mutually-exclusive options; ``log_vars`` wins):
    - weights: optional STATIC {"a↔b": w} to down-weight weakly-coupled pairs
        (e.g. star↔gas). Keys are matched in either order; default weight 1.0.
        Loss is the weighted mean Σ wᵢ·Lᵢ / Σ wᵢ (magnitude ≈ one InfoNCE term).
    - log_vars: optional LEARNABLE {"a↔b": s} where s = log σ² is a scalar
        Parameter per pair (Kendall et al. 2018 homoscedastic uncertainty
        weighting). Each term becomes exp(-s)·Lᵢ + s — the exp(-s) factor
        down-weights pairs the model is uncertain about, and the +s regularizer
        stops s→+∞ from trivially zeroing the loss. Returned loss is the mean of
        these terms over pairs. Missing keys default to s=0 (weight 1, no reg).

    Returns (loss, per_pair_losses_dict). per_pair_losses always holds the RAW
    (unweighted) InfoNCE value per pair, so the logged curves stay interpretable
    regardless of the weighting mode.
    """
    domains = list(embeddings.keys())
    if len(domains) < 2:
        raise ValueError(f"all_pairs loss needs >=2 domains, got {domains}")
    any_z = next(iter(embeddings.values()))
    total = any_z.new_zeros(())
    pair_losses: dict[str, float] = {}
    wsum = 0.0
    wsum_t = any_z.new_zeros(())   # tensor accumulator for tensor_weights mode
    n_pairs = 0
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            da, db = domains[i], domains[j]
            loss_ij = info_nce_loss(
                embeddings[da], embeddings[db], temperature,
                queue_a=(queues.get(da) if queues else None),
                queue_b=(queues.get(db) if queues else None),
            )
            pair_losses[f"{da}↔{db}"] = loss_ij.item()
            n_pairs += 1
            if tensor_weights is not None:
                # Learnable weights kept as TENSORS (the `weights` path casts to
                # float and would sever the gradient to pair_log_vars). Plain
                # weighted mean; the forced-spread normalization already pins
                # the scale, so no +s regularizer is needed here.
                w_t = tensor_weights.get(f"{da}↔{db}",
                                         tensor_weights.get(f"{db}↔{da}", None))
                if w_t is None:
                    w_t = any_z.new_ones(())
                total = total + w_t * loss_ij
                wsum_t = wsum_t + w_t
            elif log_vars is not None:
                s = log_vars.get(f"{da}↔{db}", log_vars.get(f"{db}↔{da}", None))
                if s is None:
                    total = total + loss_ij
                else:
                    total = total + torch.exp(-s) * loss_ij + s
            else:
                if weights is None:
                    w = 1.0
                else:
                    w = float(weights.get(f"{da}↔{db}",
                                          weights.get(f"{db}↔{da}", 1.0)))
                total = total + w * loss_ij
                wsum += w
    if tensor_weights is not None:
        return total / wsum_t.clamp_min(1e-6), pair_losses
    denom = n_pairs if log_vars is not None else max(wsum, 1.0)
    return total / max(denom, 1.0), pair_losses


@torch.no_grad()
def cross_domain_retrieval_accuracy(embeddings: dict[str, torch.Tensor]) -> float:
    """Mean top-1 retrieval accuracy over all ordered domain pairs.

    For each pair (a, b), checks whether embedding a_i retrieves its true
    partner b_i (and vice-versa) as the nearest neighbour in the batch. This is
    a *generalization* signal that does not collapse the way a contrastive
    train loss does — a model that has merely memorized the training batch
    scores near chance (1/B) on held-out data even when its train loss is ~0.
    Computed on whatever batch is passed (typically the local val batch).
    """
    domains = list(embeddings.keys())
    accs = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            za, zb = embeddings[domains[i]], embeddings[domains[j]]
            tgt = torch.arange(za.size(0), device=za.device)
            sim = za @ zb.t()
            accs.append((sim.argmax(dim=1) == tgt).float().mean())
            accs.append((sim.t().argmax(dim=1) == tgt).float().mean())
    if not accs:
        return 0.0
    return torch.stack(accs).mean().item()
