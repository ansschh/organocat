#!/usr/bin/env python3
"""Validate the catalyst batch offset (cat_batch_from) over EVERY 64-pair chunk
on CPU, so we can confirm the fix without burning a GPU job. Reports any chunk
whose batched catalyst edge_index exceeds the catalyst node count."""
import sys, os, pickle
ROOT = os.environ.get("ORGANOCAT_ROOT", ".")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch
from torch_geometric.data import Batch
from run_zcat_bench import as_pairdata, cat_batch_from

pairs = pickle.load(open("data/pairs/contrastive_pairs.pt", "rb"))["pairs"]
print("loaded", len(pairs), flush=True)
as_pairdata(pairs)          # sets PairData + cat_n_bonds

bad = total = 0
for i in range(0, len(pairs), 64):
    chunk = pairs[i:i + 64]; total += 1
    b = Batch.from_data_list(chunk)
    cat = cat_batch_from(b)
    npos = cat.pos.size(0)
    emax = int(cat.edge_index.max()) if cat.edge_index.numel() else -1
    nb = int(cat.batch.max()) + 1
    mm = int(cat.metal_mask.sum())
    if emax >= npos or nb != len(chunk) or mm != len(chunk):
        bad += 1
        if bad <= 5:
            print(f"BAD chunk @ {i}: edge_max={emax} npos={npos} "
                  f"nbatch={nb} metal_sum={mm} chunk={len(chunk)}", flush=True)
print(f"bad chunks: {bad}/{total}", flush=True)
print("FIX OK — catalyst batching is correct" if bad == 0
      else "STILL BROKEN", flush=True)
