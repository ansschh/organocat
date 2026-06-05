#!/usr/bin/env python3
"""Run the tmQM parser on the full dataset and pickle the result for fast reload.

Output: data/tmqm/tmqm_full.pkl  — dict CSD_code -> complex_dict
"""
from __future__ import annotations
import os, pickle, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.tmqm_parser import load_tmqm, summarize


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmqm-dir", default="data/raw/tmQM/tmQM")
    ap.add_argument("--out", default="data/tmqm/tmqm_full.pkl")
    args = ap.parse_args()

    t0 = time.time()
    complexes = load_tmqm(args.tmqm_dir, limit=None)
    print(f"\nload time: {(time.time()-t0):.1f} sec")
    summarize(complexes)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(complexes, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\nwrote {args.out}  ({sz_mb:.1f} MB)  containing {len(complexes)} complexes")


if __name__ == "__main__":
    main()
