#!/usr/bin/env python3
"""Train/val/test splits for Z_cat-CLIP.

The retrieval-leakage lesson: random splits lie. Oracle retrieval "worked"
only because it used privileged pocket coordinates; once removed, chemistry-only
retrieval collapsed to generic heme-like rankings. So we evaluate under splits
that force generalization to genuinely unseen chemistry:

  - leave_catalyst_out : val/test see catalyst SMILES never seen in train
  - leave_metal_out    : val/test metals (e.g. Ir, Ru) absent from train
  - leave_ligand_out   : val/test ligand classes absent from train

Each returns (train_idx, val_idx, test_idx) into the pairs list.
"""
from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple

import torch

from src.data.catalyst_descriptors import catalyst_descriptors


def _grouped_split(pairs, key_fn, train_frac, val_frac, seed, name):
    """Generic 'leave-group-out' split: partition the *groups* (not the pairs)
    into train/val/test, so a group's pairs never straddle the boundary."""
    rng = torch.Generator().manual_seed(seed)
    by_key = defaultdict(list)
    for i, p in enumerate(pairs):
        k = key_fn(p)
        if k is None:
            continue
        by_key[k].append(i)
    keys = sorted(by_key.keys())
    perm = torch.randperm(len(keys), generator=rng).tolist()
    n_tr = int(train_frac * len(keys))
    n_va = int(val_frac * len(keys))
    tr_keys = {keys[i] for i in perm[:n_tr]}
    va_keys = {keys[i] for i in perm[n_tr:n_tr + n_va]}
    te_keys = {keys[i] for i in perm[n_tr + n_va:]}
    tr, va, te = [], [], []
    for k, idxs in by_key.items():
        bucket = tr if k in tr_keys else va if k in va_keys else te
        bucket.extend(idxs)
    print(f"[{name}] groups: train={len(tr_keys)} val={len(va_keys)} test={len(te_keys)}  "
          f"| pairs: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    return tr, va, te


def _metal_of(p):
    d = catalyst_descriptors(p.catalyst_smi)
    return d["metal"] if d else None


def _ligand_of(p):
    d = catalyst_descriptors(p.catalyst_smi)
    return d["ligand_class"] if d else None


def make_split(pairs, kind: str = "leave_catalyst_out",
               train_frac=0.8, val_frac=0.1, seed=42) -> Tuple[List[int], List[int], List[int]]:
    if kind == "leave_catalyst_out":
        return _grouped_split(pairs, lambda p: p.catalyst_smi, train_frac, val_frac, seed, kind)
    if kind == "leave_metal_out":
        # metals are few; use a smaller train fraction so val/test get whole metals
        return _grouped_split(pairs, _metal_of, train_frac, val_frac, seed, kind)
    if kind == "leave_ligand_out":
        return _grouped_split(pairs, _ligand_of, train_frac, val_frac, seed, kind)
    raise ValueError(f"unknown split kind: {kind}")


ALL_SPLITS = ["leave_catalyst_out", "leave_metal_out", "leave_ligand_out"]
