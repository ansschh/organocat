#!/usr/bin/env python3
"""Find the catalyst-graph inconsistency that triggers the CUDA index assert.

Replays the bench's exact batching (PairData, chunks of 64) on CPU and reports
the first chunk whose batched cat_edge_index exceeds the catalyst node count,
plus the offending pair. Distinguishes a per-pair data bug (sanitize fixes it)
from a PairData batch-offset bug (needs a code fix).
"""
import sys, pickle
sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from collections import Counter

import torch
from torch_geometric.data import Data, Batch


class PairData(Data):
    def __inc__(self, key, value, *a, **k):
        if key in ("cat_edge_index", "cat_metal_idx"):
            return self.cat_pos.size(0)
        return super().__inc__(key, value, *a, **k)

    def __cat_dim__(self, key, value, *a, **k):
        if key == "cat_edge_index":
            return -1
        return super().__cat_dim__(key, value, *a, **k)


pairs = pickle.load(open("data/pairs/contrastive_pairs.pt", "rb"))["pairs"]
print("loaded", len(pairs), flush=True)
for p in pairs:
    p.__class__ = PairData

# 1) per-pair consistency scan
reasons = Counter(); examples = {}
for idx, p in enumerate(pairs):
    n = int(p.cat_n_atoms); ce = p.cat_edge_index; re = p.edge_index
    r = None
    if p.cat_pos.size(0) != n: r = "cat_pos!=n"
    elif int(p.cat_metal_mask.sum()) != 1: r = "metal_mask!=1"
    elif ce.numel() and (int(ce.max()) >= n or int(ce.min()) < 0): r = "cat_edge_oob"
    elif p.x.size(0) == 0: r = "empty_rxn"
    elif re.numel() and (int(re.max()) >= p.x.size(0) or int(re.min()) < 0): r = "rxn_edge_oob"
    if r:
        reasons[r] += 1
        examples.setdefault(r, (idx, p.cat_pos.size(0), n,
                                int(ce.max()) if ce.numel() else -1,
                                int(p.cat_metal_mask.sum()), p.x.size(0)))
print("per-pair dropped reasons:", dict(reasons), flush=True)
for r, ex in examples.items():
    print(f"  {r}: idx={ex[0]} cat_pos={ex[1]} cat_n={ex[2]} cat_edge_max={ex[3]} "
          f"metal_sum={ex[4]} x={ex[5]}", flush=True)

# 2) batch-offset scan (the bench batches in chunks of 64)
print("--- batch-offset scan (chunks of 64) ---", flush=True)
found = False
for i in range(0, len(pairs), 64):
    chunk = pairs[i:i + 64]
    b = Batch.from_data_list(chunk)
    npos = b.cat_pos.size(0)
    emax = int(b.cat_edge_index.max()) if b.cat_edge_index.numel() else -1
    mmask = int(b.cat_metal_mask.sum())
    if emax >= npos or mmask != len(chunk):
        found = True
        print(f"BATCH PROBLEM at chunk start {i}: cat_edge_max={emax} npos={npos} "
              f"metal_sum={mmask} chunk={len(chunk)}", flush=True)
        for p in chunk:
            n = int(p.cat_n_atoms)
            em = int(p.cat_edge_index.max()) if p.cat_edge_index.numel() else -1
            if p.cat_pos.size(0) != n or em >= n or int(p.cat_metal_mask.sum()) != 1:
                print(f"   culprit: {p.catalyst_smi[:60]} cat_pos={p.cat_pos.size(0)} "
                      f"n={n} edge_max={em} metal_sum={int(p.cat_metal_mask.sum())}", flush=True)
        break
if not found:
    print("no batch-offset problem found in any 64-chunk — PairData batching is OK", flush=True)
