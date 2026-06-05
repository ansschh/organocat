#!/usr/bin/env python3
"""Build (reaction, catalyst-3d) pairs for contrastive training. v2: per-molecule timeout.

Input:  data/reactions/ord_tm_clean.pkl  - cleaned atom-mapped reactions
Output: torch_geometric Data objects, one per reaction, containing:
  reaction graph:    .x, .edge_index, .edge_attr  (from reaction_gnn.parse_reaction_smiles)
  catalyst graph:    .cat_pos, .cat_z, .cat_charges, .cat_edge_index,
                     .cat_edge_attr, .cat_metal_mask, .cat_metal_idx
                     (built by embedding the catalyst SMILES via rdkit ETKDG)

For each unique catalyst SMILES, we 3D-embed ONCE and cache the result.
Per-molecule SIGALRM timeout prevents ETKDG hangs on weird organometallic SMILES.
"""
from __future__ import annotations
import os, pickle, signal, sys, time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.data.tmqm_dataset import ELEMENT_TO_Z
from src.data.ord_parser import TM_SYMBOLS
from src.encoders.reaction_gnn import parse_reaction_smiles

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# rdkit 2024+/2026 moved ETKDGv3 + EmbedMolecule into rdDistGeom and may not
# re-export them on AllChem. Resolve from the canonical module, fall back to
# AllChem for older rdkit. (On rdkit 2026 AllChem.ETKDGv3 can be missing, which
# made every catalyst embed raise AttributeError -> caught -> None -> 0 pairs.)
try:
    from rdkit.Chem import rdDistGeom as _DG
    _ETKDGv3 = _DG.ETKDGv3
    _EmbedMolecule = _DG.EmbedMolecule
except Exception:  # pragma: no cover
    _ETKDGv3 = AllChem.ETKDGv3
    _EmbedMolecule = AllChem.EmbedMolecule


class TimeoutError_(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def _handler(signum, frame):
        raise TimeoutError_("embed timed out")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def embed_catalyst_3d(smi: str, max_attempts: int = 3) -> Optional[dict]:
    """Generate 3D coords + bond graph for a catalyst SMILES via rdkit ETKDG."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        mol = Chem.AddHs(mol)
    except Exception:
        return None
    if mol.GetNumAtoms() > 150: return None

    params = _ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True   # critical for hard TM-complex cases
    # NB: do NOT set params.maxAttempts — not a valid ETKDG attribute on rdkit
    # 2026 (raises AttributeError); the seed-varying retry loop covers attempts.
    code = _EmbedMolecule(mol, params)
    if code != 0:
        for attempt in range(max_attempts):
            params.randomSeed = attempt * 7 + 1
            code = _EmbedMolecule(mol, params)
            if code == 0: break
    if code != 0:
        return None
    conf = mol.GetConformer()
    has_tm = any(a.GetSymbol() in TM_SYMBOLS for a in mol.GetAtoms())
    if not has_tm:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        atoms.append({"element": atom.GetSymbol(), "x": p.x, "y": p.y, "z": p.z,
                       "charge": float(atom.GetFormalCharge())})
    bonds = []
    for b in mol.GetBonds():
        bonds.append({"i": b.GetBeginAtomIdx() + 1, "j": b.GetEndAtomIdx() + 1,
                       "order": b.GetBondTypeAsDouble()})
    metal_idx = next((i for i, a in enumerate(atoms) if a["element"] in TM_SYMBOLS), None)
    if metal_idx is None: return None
    metal_elem = atoms[metal_idx]["element"]
    return {
        "smiles": smi, "n_atoms": len(atoms), "atoms": atoms, "bonds": bonds,
        "metal_idx": metal_idx, "metal_element": metal_elem,
    }


def embed_with_timeout(smi: str, timeout_s: int = 15) -> Optional[dict]:
    try:
        with time_limit(timeout_s):
            return embed_catalyst_3d(smi)
    except TimeoutError_:
        return None
    except Exception:
        return None


def catalyst_to_pyg_subgraph(cat: dict) -> Optional[dict]:
    if cat is None: return None
    atoms = cat["atoms"]; n = len(atoms)
    pos = torch.tensor([[a["x"], a["y"], a["z"]] for a in atoms], dtype=torch.float)
    z = torch.tensor([ELEMENT_TO_Z.get(a["element"], 0) for a in atoms], dtype=torch.long)
    charges = torch.tensor([a["charge"] for a in atoms], dtype=torch.float)
    src, dst, bo = [], [], []
    for b in cat["bonds"]:
        i, j = b["i"] - 1, b["j"] - 1
        if i < 0 or j < 0 or i >= n or j >= n: continue
        src += [i, j]; dst += [j, i]
        bo += [b["order"], b["order"]]
    if not src: return None
    metal_mask = torch.zeros(n, dtype=torch.bool); metal_mask[cat["metal_idx"]] = True
    return {
        "cat_pos": pos, "cat_z": z, "cat_charges": charges,
        "cat_edge_index": torch.tensor([src, dst], dtype=torch.long),
        "cat_edge_attr": torch.tensor(bo, dtype=torch.float).unsqueeze(-1),
        "cat_metal_mask": metal_mask,
        "cat_metal_idx": torch.tensor([cat["metal_idx"]], dtype=torch.long),
        "cat_n_atoms": n,
    }


def build_contrastive_pairs(clean_pkl: str, out_pkl: str,
                             cat_cache_pkl: str = "data/pairs/cat_cache.pkl",
                             limit: Optional[int] = None,
                             min_reaction_confidence: float = 0.5,
                             timeout_s: int = 15):
    with open(clean_pkl, "rb") as f:
        clean = pickle.load(f)
    if limit: clean = clean[:limit]
    print(f"loaded {len(clean)} cleaned + atom-mapped reactions", flush=True)

    # Count catalyst frequencies; rare catalysts get skipped if they fail.
    from collections import Counter
    cat_freq = Counter(r["catalyst_smi"] for r in clean)
    print(f"unique catalyst SMILES: {len(cat_freq)}", flush=True)

    # Resume from cache if present
    if os.path.exists(cat_cache_pkl):
        cat_cache = pickle.load(open(cat_cache_pkl, "rb"))
        print(f"resumed cat_cache: {len(cat_cache)} entries", flush=True)
    else:
        cat_cache = {}

    # retry entries that previously failed (cached as None), not just unseen ones
    todo = [(smi, cnt) for smi, cnt in cat_freq.most_common()
            if cat_cache.get(smi) is None]
    print(f"to embed: {len(todo)} (sorted by usage)", flush=True)

    n_ok = sum(1 for v in cat_cache.values() if v is not None)
    n_fail = sum(1 for v in cat_cache.values() if v is None)
    t0 = time.time()
    for i, (smi, cnt) in enumerate(todo):
        t_one = time.time()
        cat = embed_with_timeout(smi, timeout_s=timeout_s)
        sub = catalyst_to_pyg_subgraph(cat) if cat is not None else None
        cat_cache[smi] = sub
        if sub is not None: n_ok += 1
        else: n_fail += 1
        dt = time.time() - t_one
        if dt > 5.0:
            print(f"  slow embed ({dt:.1f}s) on cat used {cnt}x: {smi[:80]}", flush=True)
        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            elapsed = time.time() - t0
            print(f"  embedded {i+1}/{len(todo)}  ok={n_ok}  fail={n_fail}  ({elapsed:.0f}s)", flush=True)
            # Checkpoint every 20
            os.makedirs(os.path.dirname(cat_cache_pkl), exist_ok=True)
            with open(cat_cache_pkl + ".tmp", "wb") as f:
                pickle.dump(cat_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(cat_cache_pkl + ".tmp", cat_cache_pkl)
    print(f"3D-embed final: ok={n_ok} fail={n_fail}", flush=True)

    # Build pair list
    pairs = []
    n_no_cat = 0; n_bad_rxn = 0; n_lowconf = 0
    for r in clean:
        if r.get("mapping_confidence", 1.0) < min_reaction_confidence:
            n_lowconf += 1; continue
        cat_data = cat_cache.get(r["catalyst_smi"])
        if cat_data is None: n_no_cat += 1; continue
        rxn_data = parse_reaction_smiles(r.get("mapped_rxn") or f"{r['reactants_smi']}>>{r['products_smi']}")
        if rxn_data is None: n_bad_rxn += 1; continue
        combined = Data(
            x=rxn_data.x, edge_index=rxn_data.edge_index, edge_attr=rxn_data.edge_attr,
            **cat_data,
        )
        combined.reaction_id = r.get("reaction_id")
        combined.catalyst_smi = r["catalyst_smi"]
        combined.mapping_confidence = r.get("mapping_confidence", 1.0)
        combined.n_rxn_atoms = rxn_data.n_nodes
        pairs.append(combined)

    print(f"\nfinal pairs: {len(pairs)}", flush=True)
    print(f"skipped: low_confidence={n_lowconf}, no_cat={n_no_cat}, bad_reaction={n_bad_rxn}", flush=True)

    n_cat_used = len({p.catalyst_smi for p in pairs})
    print(f"unique catalysts represented in pairs: {n_cat_used}", flush=True)

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump({"pairs": pairs, "catalyst_cache_size": len(cat_cache)}, f,
                     protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved -> {out_pkl}", flush=True)
    return pairs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-pkl", default="data/reactions/ord_tm_clean.pkl")
    ap.add_argument("--out-pkl", default="data/pairs/contrastive_pairs.pt")
    ap.add_argument("--cat-cache-pkl", default="data/pairs/cat_cache.pkl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout-s", type=int, default=15)
    args = ap.parse_args()
    build_contrastive_pairs(args.clean_pkl, args.out_pkl,
                             cat_cache_pkl=args.cat_cache_pkl,
                             limit=args.limit, timeout_s=args.timeout_s)
