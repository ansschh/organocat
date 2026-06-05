#!/usr/bin/env python3
"""Train the InfoNCE contrastive model end-to-end.

Setup:
  - Load contrastive pairs (reaction graph + catalyst 3D graph).
  - Split train/val/test (catalyst-stratified — held-out catalyst SMILES test
    generalization to UNSEEN catalysts).
  - For each batch:
       reaction_emb  = projection( reaction_gnn(rxn_data) )
       catalyst_emb  = projection( complex_egnn(cat_data) )
       loss = info_nce(reaction_emb, catalyst_emb)
  - Eval: top-k retrieval accuracy on held-out reactions (do they retrieve
    the correct catalyst from the full catalyst pool?)
"""
from __future__ import annotations
import os, pickle, sys, time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.encoders.reaction_gnn import ReactionGNN
from src.encoders.complex_egnn import ComplexEGNN
from src.contrastive.info_nce import info_nce_loss, retrieval_metrics, ProjectionHead


class ReactionView(nn.Module):
    """Wrap a reaction graph Data into something ComplexEGNN/ReactionGNN-friendly."""

    @staticmethod
    def rxn_batch_from(combined_batch):
        # The combined Data has x, edge_index, edge_attr (reaction graph) and cat_* (catalyst)
        # ReactionGNN reads .x, .edge_index, .edge_attr, .batch — these are already there.
        return combined_batch

    @staticmethod
    def cat_batch_from(combined_batch):
        """Build a separate Batch-compatible object for the catalyst encoder.
        The combined batch has cat_pos, cat_z, etc. as concatenated tensors.
        We need a separate batch index for the catalyst nodes."""
        from torch_geometric.data import Data
        # Build per-graph catalyst tensors and re-batch
        # The cat_* tensors are already concatenated across the batch.
        # The catalyst batch index is computed from cat_n_atoms.
        device = combined_batch.cat_pos.device
        n_per_graph = combined_batch.cat_n_atoms     # tensor or list
        if torch.is_tensor(n_per_graph):
            n_per_graph_list = n_per_graph.tolist()
        else:
            n_per_graph_list = list(n_per_graph)
        cat_batch = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=device)
            for i, n in enumerate(n_per_graph_list)
        ])
        # Construct a faux Data with the right field names
        cat = Data(
            pos=combined_batch.cat_pos, z=combined_batch.cat_z,
            charges=combined_batch.cat_charges,
            edge_index=combined_batch.cat_edge_index,
            edge_attr=combined_batch.cat_edge_attr,
            metal_mask=combined_batch.cat_metal_mask,
        )
        cat.batch = cat_batch
        cat.metal_idx = combined_batch.cat_metal_idx
        return cat


def split_pairs_by_catalyst(pairs, train_frac=0.8, val_frac=0.1, seed=42):
    """Catalyst-stratified split: train sees a subset of catalysts; val/test
    sees DIFFERENT catalysts. This tests retrieval generalization to unseen
    catalysts (the real Z_cat test).
    """
    rng = torch.Generator().manual_seed(seed)
    by_cat = defaultdict(list)
    for i, p in enumerate(pairs):
        by_cat[p.catalyst_smi].append(i)
    cats = sorted(by_cat.keys())
    perm = torch.randperm(len(cats), generator=rng).tolist()
    n_train_cat = int(train_frac * len(cats))
    n_val_cat = int(val_frac * len(cats))
    train_cats = {cats[i] for i in perm[:n_train_cat]}
    val_cats = {cats[i] for i in perm[n_train_cat:n_train_cat + n_val_cat]}
    test_cats = {cats[i] for i in perm[n_train_cat + n_val_cat:]}
    train_idx, val_idx, test_idx = [], [], []
    for i, p in enumerate(pairs):
        if p.catalyst_smi in train_cats: train_idx.append(i)
        elif p.catalyst_smi in val_cats: val_idx.append(i)
        else: test_idx.append(i)
    print(f"split (catalyst-stratified): train_cats={len(train_cats)} val_cats={len(val_cats)} "
          f"test_cats={len(test_cats)}")
    print(f"  train pairs={len(train_idx)} val pairs={len(val_idx)} test pairs={len(test_idx)}")
    return train_idx, val_idx, test_idx


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-pkl", default="data/pairs/contrastive_pairs.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on number of pairs (for fast iteration)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"=== loading pairs ===")
    bundle = pickle.load(open(args.pairs_pkl, "rb"))
    pairs = bundle["pairs"]
    if args.limit: pairs = pairs[:args.limit]
    print(f"loaded {len(pairs)} pairs")

    train_idx, val_idx, test_idx = split_pairs_by_catalyst(pairs)
    train_ds = [pairs[i] for i in train_idx]
    val_ds   = [pairs[i] for i in val_idx]
    test_ds  = [pairs[i] for i in test_idx]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"=== building model ===")
    rxn_enc = ReactionGNN(embed_dim=args.embed_dim).to(args.device)
    cat_enc = ComplexEGNN(embed_dim=args.embed_dim, update_coords=False).to(args.device)
    proj_r = ProjectionHead(args.embed_dim, args.embed_dim).to(args.device)
    proj_c = ProjectionHead(args.embed_dim, args.embed_dim).to(args.device)

    n_params = sum(p.numel() for p in rxn_enc.parameters()) + \
               sum(p.numel() for p in cat_enc.parameters()) + \
               sum(p.numel() for p in proj_r.parameters()) + \
               sum(p.numel() for p in proj_c.parameters())
    print(f"  total params: {n_params/1e6:.2f} M")

    params = list(rxn_enc.parameters()) + list(cat_enc.parameters()) + \
             list(proj_r.parameters()) + list(proj_c.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    def encode(batch):
        batch = batch.to(args.device)
        h_r = rxn_enc(batch)
        cat = ReactionView.cat_batch_from(batch)
        h_c = cat_enc(cat)
        return proj_r(h_r), proj_c(h_c)

    def epoch_eval(loader):
        rxn_enc.eval(); cat_enc.eval(); proj_r.eval(); proj_c.eval()
        all_r, all_c = [], []
        with torch.no_grad():
            for batch in loader:
                z_r, z_c = encode(batch)
                all_r.append(z_r); all_c.append(z_c)
        if not all_r: return {}
        all_r = torch.cat(all_r); all_c = torch.cat(all_c)
        return retrieval_metrics(all_r, all_c, top_ks=(1, 5, 10))

    print(f"=== training {args.epochs} epochs ===")
    for epoch in range(args.epochs):
        rxn_enc.train(); cat_enc.train(); proj_r.train(); proj_c.train()
        t0 = time.time()
        running_loss = 0; running_n = 0
        for batch in train_loader:
            z_r, z_c = encode(batch)
            loss = info_nce_loss(z_r, z_c, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            running_loss += loss.item() * batch.num_graphs
            running_n += batch.num_graphs
        train_loss = running_loss / running_n
        val_metrics = epoch_eval(val_loader)
        dt = time.time() - t0
        print(f"epoch {epoch+1:2d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val: hits@1={val_metrics.get('hits@1', 0):.3f}  hits@5={val_metrics.get('hits@5', 0):.3f}  "
              f"mrr={val_metrics.get('mrr', 0):.3f}  ({dt:.0f}s)")

    print(f"\n=== final test (UNSEEN catalysts) ===")
    test_metrics = epoch_eval(test_loader)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
