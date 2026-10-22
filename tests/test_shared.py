"""Sanity tests for the modules copied from GalaxyCLIP-2D. Run: python tests/test_shared.py"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.losses import all_pairs_info_nce_loss, cross_domain_retrieval_accuracy
from galaxyclip3d.queue import EmbeddingQueue
from galaxyclip3d.training import ZClipState, full_gallery_recall, ModelEMA, queue_size_at_epoch
from galaxyclip3d.ids import load_2d_blocks, load_pairing, halo_of_2d_sample, blocks_by_halo

torch.manual_seed(0)
doms = ["gas", "dm", "star"]
base = F.normalize(torch.randn(64, 32), dim=-1)
aligned = {d: base.clone() for d in doms}
rand = {d: F.normalize(torch.randn(64, 32), dim=-1) for d in doms}

l_al, _ = all_pairs_info_nce_loss(aligned, 0.07)
l_rn, _ = all_pairs_info_nce_loss(rand, 0.07)
assert torch.isfinite(l_al) and l_al < l_rn, (l_al, l_rn)
assert abs(cross_domain_retrieval_accuracy(aligned) - 1.0) < 1e-6
print(f"PASS losses: aligned {l_al.item():.4f} < random {l_rn.item():.4f}; aligned retrieval acc = 1")

q = EmbeddingQueue(doms, 16)
assert q.negatives() == {}
q.enqueue({d: v[:10] for d, v in aligned.items()})
assert q.negatives()["gas"].shape == (10, 32)
q.enqueue({d: v[10:20] for d, v in aligned.items()})
assert q.negatives()["dm"].shape == (16, 32)
assert q.set_queue_size(24) and q.negatives()["star"].shape == (16, 32)
l_q, _ = all_pairs_info_nce_loss(aligned, 0.07, queues=q.negatives())
assert torch.isfinite(l_q)
print("PASS queue: fills to K, resize keeps entries, loss accepts queued negatives")

full = full_gallery_recall(aligned, (1, 5))
assert abs(full["r@1"] - 1.0) < 1e-6
print(f"PASS full_gallery_recall: R@1 = {full['r@1']:.3f} on identical embeddings")

z = ZClipState(z=2.5, alpha=0.97, warmup_steps=5)
for v in [1.0, 1.1, 0.9, 1.0, 1.05]:
    z.update(v, z.threshold())
spike = z.update(50.0, z.threshold())
assert z.ready and spike
print(f"PASS ZClip: arms after warm-up, flags a spike (threshold {z.threshold():.3f})")

live, shadow = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
ema = ModelEMA(live, shadow, decay=0.5)
w0 = live.weight.detach().clone()
with torch.no_grad():
    live.weight.add_(1.0)
ema.update()
assert torch.allclose(ema.model.weight, w0 + 0.5)
print("PASS ModelEMA: ema = d*ema + (1-d)*live")

class C: queue_size = 0; queue_start_epoch = 0; queue_ramp_every = 10; queue_ramp_step = 64; queue_ramp_max = 0; queue_size_2 = 0; queue_start_epoch_2 = 0
assert [queue_size_at_epoch(C, e) for e in (0, 9, 10, 25)] == [0, 0, 64, 128]
print("PASS queue_size_at_epoch: +64 every 10 epochs")

blocks = load_2d_blocks()
pairs = load_pairing("domain_translation")
assert len(blocks) == 18000 and len(pairs) == 1000 and all(p["central"] for p in pairs)
bh = blocks_by_halo(blocks, 99)
assert all(p["halo"] in bh for p in pairs)
assert halo_of_2d_sample(bh[pairs[0]["halo"]][0] * 4 + 3, blocks) == (99, pairs[0]["halo"])
print(f"PASS ids: 18000 2D blocks, 1000 pairable 3D centrals, sample->halo lookup consistent")
print("ALL TESTS PASSED")
