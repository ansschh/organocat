#!/usr/bin/env python3
"""Sanity-check pretraining: predict HOMO/LUMO/gap from complex 3D structure.

Goal: verify the ComplexEGNN can learn USEFUL features. If MAE on held-out
HL_gap << gap-stddev-of-dataset, the encoder works. If not, the encoder is
broken and contrastive learning won't fix it.

This is the cheapest meaningful sanity check before any contrastive work.
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.tmqm_dataset import TMQMDataset
from src.encoders.complex_egnn import ComplexEGNN, RegressionHead


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="data/tmqm/tmqm_full.pkl")
    ap.add_argument("--metals", nargs="*", default=None)
    ap.add_argument("--max-atoms", type=int, default=120)   # ~80% of dataset
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-targets", type=int, default=6,
                    help="number of regression targets (matches dataset.target_keys)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"=== loading dataset ===")
    ds = TMQMDataset(args.pickle, metals=args.metals, max_atoms=args.max_atoms)
    n = len(ds)
    n_train = int(0.8 * n); n_val = int(0.1 * n); n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)
    print(f"  train={n_train} val={n_val} test={n_test}")

    # Compute target normalization from training set
    print(f"=== computing target stats ===")
    ys = torch.stack([ds[i].y for i in train_ds.indices], dim=0).squeeze(1)  # (n_train, T)
    # filter NaN per target
    mean = torch.nanmean(ys, dim=0)
    std = torch.tensor([
        (ys[:, i][~torch.isnan(ys[:, i])]).std() for i in range(ys.size(1))
    ])
    std = torch.clamp(std, min=1e-3)
    print(f"  y mean: {mean.tolist()}")
    print(f"  y std:  {std.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"=== building model ===")
    encoder = ComplexEGNN(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                          embed_dim=args.embed_dim).to(args.device)
    head = RegressionHead(embed_dim=args.embed_dim, n_targets=args.n_targets).to(args.device)
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  params: {n_params/1e6:.2f} M")

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    mean_d = mean.to(args.device); std_d = std.to(args.device)

    def step(batch, train=True):
        batch = batch.to(args.device)
        emb = encoder(batch)
        pred = head(emb)
        target = batch.y                              # (B, T)
        # Normalize, mask NaN
        target_norm = (target - mean_d) / std_d
        mask = ~torch.isnan(target_norm)
        loss = F.mse_loss(pred[mask], target_norm[mask])
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 5.0)
            opt.step()
        # Per-target MAE in original units
        with torch.no_grad():
            pred_orig = pred * std_d + mean_d
            mae_per_target = torch.where(mask, (pred_orig - target).abs(), torch.zeros_like(target)).sum(dim=0) / mask.float().sum(dim=0).clamp(min=1)
        return loss.item(), mae_per_target.detach().cpu()

    print(f"=== training {args.epochs} epochs ===")
    target_keys = ds.target_keys
    for epoch in range(args.epochs):
        encoder.train(); head.train()
        t0 = time.time()
        train_loss = 0; train_n = 0
        for batch in train_loader:
            loss, _ = step(batch, train=True)
            train_loss += loss * batch.num_graphs
            train_n += batch.num_graphs
        train_loss /= train_n

        encoder.eval(); head.eval()
        val_loss = 0; val_n = 0
        val_mae_acc = None
        with torch.no_grad():
            for batch in val_loader:
                loss, mae = step(batch, train=False)
                val_loss += loss * batch.num_graphs
                val_n += batch.num_graphs
                val_mae_acc = mae if val_mae_acc is None else val_mae_acc + mae * batch.num_graphs / args.batch_size
        val_loss /= val_n
        val_mae = (val_mae_acc / (val_n / args.batch_size)).tolist() if val_mae_acc is not None else []

        dt = time.time() - t0
        mae_str = "  ".join(f"{k}={v:.3f}" for k, v in zip(target_keys, val_mae))
        print(f"epoch {epoch+1:2d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({dt:.0f}s)  val_MAE: {mae_str}")

    # Final test
    encoder.eval(); head.eval()
    test_mae_acc = None
    test_n = 0
    with torch.no_grad():
        for batch in test_loader:
            _, mae = step(batch, train=False)
            test_mae_acc = mae if test_mae_acc is None else test_mae_acc + mae * batch.num_graphs / args.batch_size
            test_n += batch.num_graphs
    test_mae = (test_mae_acc / (test_n / args.batch_size)).tolist()
    print(f"\n=== test MAE ===")
    for k, v, s in zip(target_keys, test_mae, std.tolist()):
        print(f"  {k:>16}  MAE={v:.4f}  std={s:.4f}  MAE/std={v/s:.3f}")


if __name__ == "__main__":
    main()
