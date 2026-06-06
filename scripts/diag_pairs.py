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
pairs = as_pairdata(pairs)          # genuine PairData + cat_n_bonds
p0 = pairs[0]
print("type:", type(p0).__name__,
      "| inc(cat_edge_index) =", p0.__inc__("cat_edge_index", p0.cat_edge_index),
      "(want 0)", flush=True)

bad = total = 0
for i in range(0, len(pairs), 64):
    chunk = pairs[i:i + 64]; total += 1
    b = Batch.from_data_list(chunk)
    # reaction graph: edges must index into x, batch must cover x and span chunk
    r_emax = int(b.edge_index.max()) if b.edge_index.numel() else -1
    r_n = b.x.size(0)
    r_ok = (b.batch.size(0) == r_n) and (int(b.batch.max()) + 1 == len(chunk)) and r_emax < r_n
    # catalyst graph (via the actual cat_batch_from)
    cat = cat_batch_from(b)
    c_emax = int(cat.edge_index.max()) if cat.edge_index.numel() else -1
    c_n = cat.pos.size(0)
    c_ok = (cat.batch.size(0) == c_n) and (int(cat.batch.max()) + 1 == len(chunk)) \
        and c_emax < c_n and int(cat.metal_mask.sum()) == len(chunk)
    if not (r_ok and c_ok):
        bad += 1
        if bad <= 5:
            print(f"BAD @ {i}: rxn[emax={r_emax}/{r_n} ok={r_ok}] "
                  f"cat[emax={c_emax}/{c_n} ok={c_ok}]", flush=True)
print(f"bad chunks: {bad}/{total}", flush=True)
print("FIX OK — reaction AND catalyst batching correct" if bad == 0
      else "STILL BROKEN", flush=True)
