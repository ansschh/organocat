#!/usr/bin/env python3
"""Baselines for Z_cat-CLIP retrieval — the anti-self-deception layer.

The doc is blunt: "Before training the fancy model, run dumb baselines. This
protects us from fooling ourselves." If learned Z_cat only beats random but not
the metal-frequency prior or Morgan-FP-kNN, it is not enough.

Every baseline (and the learned model) produces a score matrix
    S : (n_eval_reactions, n_pool_catalysts)   higher = more preferred
which feeds the SAME metric (`retrieval_from_scores`), so results are directly
comparable on every split.

Reactions are plain dicts: {reactants_smi, products_smi, catalyst_smi}. The
candidate pool is the list of unique catalyst SMILES the eval reactions draw
from (unseen catalysts under leave-catalyst-out, etc.).
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from src.data.catalyst_descriptors import catalyst_descriptors

_FP_BITS = 2048


# ----------------------------- pool + metric -----------------------------

def build_pool(eval_rxns: List[dict]) -> Tuple[List[str], torch.LongTensor]:
    """Unique catalyst pool for the eval set + target index per reaction."""
    pool = list(dict.fromkeys(r["catalyst_smi"] for r in eval_rxns))
    idx_of = {s: i for i, s in enumerate(pool)}
    target = torch.tensor([idx_of[r["catalyst_smi"]] for r in eval_rxns], dtype=torch.long)
    return pool, target


def _family(smi: str) -> Optional[Tuple[str, str]]:
    d = catalyst_descriptors(smi)
    return (d["metal"], d["ligand_class"]) if d else None


def retrieval_from_scores(scores: torch.Tensor, target: torch.LongTensor,
                          pool: List[str], top_ks=(1, 5, 10)) -> Dict[str, float]:
    """Exact + family hits@k and mrr from a (n_rxn, n_pool) score matrix.

    family-hit@k: the true catalyst's (metal, ligand_class) appears in top-k.
    This is the doc's "correct catalyst/active-state family" criterion and is
    robust to there being several equally-correct catalysts in the pool."""
    n, P = scores.shape
    order = scores.argsort(dim=-1, descending=True)               # (n, P)
    fam = [_family(s) for s in pool]
    tgt_fam = [fam[t] for t in target.tolist()]
    out = {}
    # exact
    rankpos = (order == target.unsqueeze(-1)).float().argmax(dim=-1)  # rank of true
    for k in top_ks:
        out[f"hits@{k}"] = (rankpos < k).float().mean().item()
    out["mrr"] = (1.0 / (rankpos.float() + 1)).mean().item()
    out["mean_rank"] = rankpos.float().mean().item()
    # family
    fam_arr = np.array([f"{m}|{l}" if f else "none" for f in fam for (m, l) in [f or (None, None)]])
    fam_idx = {f: i for i, f in enumerate(sorted(set(fam_arr)))}
    pool_fam_id = torch.tensor([fam_idx[f] for f in fam_arr])
    tgt_fam_id = torch.tensor([fam_idx[f"{m}|{l}" if f else 'none'] for f in tgt_fam
                               for (m, l) in [f or (None, None)]])
    ranked_fam = pool_fam_id[order]                                # (n, P)
    for k in top_ks:
        hit = (ranked_fam[:, :k] == tgt_fam_id.unsqueeze(-1)).any(dim=-1).float().mean()
        out[f"fam_hits@{k}"] = hit.item()
    return out


# ----------------------------- fingerprints -----------------------------

def _morgan(smi: str, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=_FP_BITS)

def _multi_morgan(dotted_smi: str, radius=2):
    """OR-combine fingerprints of a dotted multi-component SMILES."""
    fp = None
    for s in dotted_smi.split("."):
        f = _morgan(s, radius)
        if f is None:
            continue
        if fp is None:
            fp = f
        else:
            fp = fp | f
    return fp

def _rxn_fp(rxn: dict, mode: str):
    """Reaction fingerprint: 'reactant' = reactant FP only;
    'diff' = concat(reactant, product) bit vectors (captures Δ implicitly)."""
    rf = _multi_morgan(rxn["reactants_smi"])
    if mode == "reactant":
        return rf
    pf = _multi_morgan(rxn["products_smi"])
    if rf is None or pf is None:
        return None
    return (rf, pf)


# ----------------------------- baselines -----------------------------

def random_scores(eval_rxns, pool, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(len(eval_rxns), len(pool), generator=g)


def popularity_scores(train_rxns, eval_rxns, pool) -> torch.Tensor:
    """Metal-frequency / catalyst-popularity prior: every reaction gets the same
    ranking = training frequency of each pool catalyst. Strong because a handful
    of Pd catalysts dominate ORD."""
    freq = defaultdict(float)
    for r in train_rxns:
        freq[r["catalyst_smi"]] += 1.0
    # Back off to family frequency for pool catalysts unseen in train (the usual
    # case under leave-catalyst-out / leave-metal-out).
    fam_freq = defaultdict(float)
    for r in train_rxns:
        f = _family(r["catalyst_smi"])
        if f:
            fam_freq[f] += 1.0
    row = torch.tensor([freq.get(s, 0.0) + 1e-3 * fam_freq.get(_family(s), 0.0)
                        for s in pool], dtype=torch.float)
    return row.unsqueeze(0).expand(len(eval_rxns), -1).contiguous()


def morgan_knn_scores(train_rxns, eval_rxns, pool, mode="diff",
                      topk_neighbors=25, max_train=50000, seed=0) -> torch.Tensor:
    """kNN in reaction-fingerprint space. For each eval reaction, score pool
    catalyst c = max Tanimoto similarity to any train reaction whose catalyst
    is c (nearest-neighbor vote). 'diff' uses reactant+product; 'reactant' uses
    reactants only (ablation: does product/Δ info matter).

    Uses rdkit's C++ BulkTanimotoSimilarity (NOT a Python loop) so it scales to
    100k+ train reactions; train is capped at max_train via random sample."""
    import random
    rng = random.Random(seed)
    if len(train_rxns) > max_train:
        train_rxns = rng.sample(train_rxns, max_train)

    # Precompute train FPs as flat lists for bulk similarity.
    tr_cat = []
    tr_fps = []            # reactant-mode: list of bitvects
    tr_rfps, tr_pfps = [], []   # diff-mode: reactant + product bitvects
    for r in train_rxns:
        fp = _rxn_fp(r, mode)
        if fp is None:
            continue
        if mode == "reactant":
            tr_fps.append(fp)
        else:
            tr_rfps.append(fp[0]); tr_pfps.append(fp[1])
        tr_cat.append(r["catalyst_smi"])
    pool_idx = {s: i for i, s in enumerate(pool)}

    scores = torch.zeros(len(eval_rxns), len(pool))
    if not tr_cat:
        return scores
    for ei, r in enumerate(eval_rxns):
        ef = _rxn_fp(r, mode)
        if ef is None:
            continue
        if mode == "reactant":
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(ef, tr_fps))
        else:
            sr = np.asarray(DataStructs.BulkTanimotoSimilarity(ef[0], tr_rfps))
            sp = np.asarray(DataStructs.BulkTanimotoSimilarity(ef[1], tr_pfps))
            sims = 0.5 * (sr + sp)
        order = np.argsort(sims)[::-1][:topk_neighbors]
        for j in order:
            pi = pool_idx.get(tr_cat[j])
            if pi is None:
                continue                       # train catalyst not in eval pool
            if sims[j] > float(scores[ei, pi]):
                scores[ei, pi] = float(sims[j])
        # family back-off: if no train neighbor shares an eval-pool catalyst,
        # spread the neighbor's similarity over pool catalysts of the same family
        if float(scores[ei].max()) == 0:
            fam_best = defaultdict(float)
            for j in order:
                f = _family(tr_cat[j])
                if f:
                    fam_best[f] = max(fam_best[f], float(sims[j]))
            for pi, s in enumerate(pool):
                f = _family(s)
                if f in fam_best:
                    scores[ei, pi] = fam_best[f]
    return scores


def run_all_baselines(train_rxns, eval_rxns, pool, target, top_ks=(1, 5, 10)) -> Dict[str, Dict]:
    out = {}
    out["random"] = retrieval_from_scores(random_scores(eval_rxns, pool), target, pool, top_ks)
    out["popularity_prior"] = retrieval_from_scores(
        popularity_scores(train_rxns, eval_rxns, pool), target, pool, top_ks)
    out["morgan_knn_diff"] = retrieval_from_scores(
        morgan_knn_scores(train_rxns, eval_rxns, pool, mode="diff"), target, pool, top_ks)
    out["morgan_knn_reactant"] = retrieval_from_scores(
        morgan_knn_scores(train_rxns, eval_rxns, pool, mode="reactant"), target, pool, top_ks)
    return out
