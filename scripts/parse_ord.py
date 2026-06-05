#!/usr/bin/env python3
"""Driver: scan an ORD data directory -> ord_tm_reactions.pkl (TM-tagged).

Usage:
  # inspect the first parquet's schema (use if auto-detect fails)
  python scripts/parse_ord.py --ord-root data/raw/ord-data/data --inspect

  # full parse
  python scripts/parse_ord.py --ord-root data/raw/ord-data/data \
      --out data/reactions/ord_tm_reactions.pkl
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.data.ord_parser import parse_ord_root, find_ord_files


def inspect(ord_root: str):
    files = find_ord_files(ord_root)
    print(f"found {len(files)} ORD files")
    if not files:
        return
    p = files[0]
    print(f"first file: {p}")
    if p.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(p)
        print(f"  rows={len(df)}  columns={list(df.columns)}")
        for col in df.columns:
            s = df[col].dropna()
            t = type(s.iloc[0]).__name__ if not s.empty else "empty"
            print(f"    {col}: dtype={df[col].dtype} first_type={t}")
    else:
        print("  (.pb.gz dataset protobuf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ord-root", default="data/raw/ord-data/data")
    ap.add_argument("--out", default="data/reactions/ord_tm_reactions.pkl")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="keep ALL reactions, not just TM")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.ord_root)
        return

    rxns = parse_ord_root(args.ord_root, tm_only=not args.all,
                          limit_files=args.limit_files)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(rxns, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(rxns)} reactions -> {args.out}")


if __name__ == "__main__":
    main()
