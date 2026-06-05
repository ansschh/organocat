#!/usr/bin/env python3
"""Hard-negative sampling for Z_cat-CLIP.

The doc is explicit: "The important part is not the equation. The important
part is the negative design." Shallow geometry gives heme-like false positives,
so a useful Z_cat must separate "looks like a cofactor pocket" from "supports
this reaction." We attack that directly with chemically adversarial negatives.

Categories (mapped from the doc's six):
  1. same_metal_diff_ligand     - same metal, wrong ligand field
  2. diff_metal                 - wrong metal (same/any reaction)
  3. same_metal_same_ligand     - correct family, different instance
                                  (proxy for wrong oxidation / active state)
  4. shape_match_diff_chem      - similar size + donor profile, different
                                  metal-or-ligand (the "geometrically tempting
                                  but chemically wrong" / heme-like analog)

A HardNegativeSampler is built over a *catalyst bank* (the unique catalysts in
the training split, each with a 3D subgraph + cheap descriptors). At batch time
it returns, for each anchor catalyst, K bank indices to use as negatives. The
training driver encodes the bank once per step and gathers the negative
embeddings into the (B, K, D) tensor that info_nce_loss already accepts.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from src.data.catalyst_descriptors import catalyst_descriptors

CATEGORIES = ["same_metal_diff_ligand", "diff_metal",
              "same_metal_same_ligand", "shape_match_diff_chem"]


class CatalystBank:
    """The pool of unique catalysts available as negatives, with descriptors
    and an index over (metal, ligand_class) for fast category sampling."""

    def __init__(self, smis: List[str]):
        self.smis = list(dict.fromkeys(smis))           # unique, order-preserving
        self.idx_of = {s: i for i, s in enumerate(self.smis)}
        self.desc: List[Optional[Dict]] = [catalyst_descriptors(s) for s in self.smis]

        self.by_metal = defaultdict(list)
        self.by_metal_ligand = defaultdict(list)
        self.metal = [None] * len(self.smis)
        self.ligand = [None] * len(self.smis)
        self.size = torch.zeros(len(self.smis))
        self.donor = torch.zeros(len(self.smis), 4)     # [P, N, O, halide>0]
        for i, d in enumerate(self.desc):
            if d is None:
                continue
            self.metal[i] = d["metal"]; self.ligand[i] = d["ligand_class"]
            self.by_metal[d["metal"]].append(i)
            self.by_metal_ligand[(d["metal"], d["ligand_class"])].append(i)
            self.size[i] = d["n_atoms"]
            self.donor[i] = torch.tensor([float(d["has_P"]), float(d["has_N"]),
                                          float(d["has_O"]), float(d["n_halide"] > 0)])
        self.all_idx = [i for i, d in enumerate(self.desc) if d is not None]

    def __len__(self):
        return len(self.smis)


class HardNegativeSampler:
    def __init__(self, bank: CatalystBank, k: int = 8, seed: int = 0,
                 weights: Optional[Dict[str, float]] = None):
        self.bank = bank
        self.k = k
        self.g = torch.Generator().manual_seed(seed)
        # Default mix across the four categories.
        w = weights or {"same_metal_diff_ligand": 0.35, "diff_metal": 0.25,
                        "same_metal_same_ligand": 0.15, "shape_match_diff_chem": 0.25}
        self.cats = list(w.keys())
        self.cat_p = torch.tensor([w[c] for c in self.cats], dtype=torch.float)

    def _rand_choice(self, pool: List[int]) -> Optional[int]:
        if not pool:
            return None
        j = torch.randint(len(pool), (1,), generator=self.g).item()
        return pool[j]

    def _sample_one(self, anchor_smi: str, category: str) -> Optional[int]:
        b = self.bank
        ai = b.idx_of.get(anchor_smi)
        if ai is None or b.desc[ai] is None:
            return self._rand_choice(b.all_idx)
        metal, ligand = b.metal[ai], b.ligand[ai]

        if category == "same_metal_diff_ligand":
            pool = [i for i in b.by_metal.get(metal, []) if b.ligand[i] != ligand]
        elif category == "diff_metal":
            pool = [i for i in b.all_idx if b.metal[i] != metal]
        elif category == "same_metal_same_ligand":
            pool = [i for i in b.by_metal_ligand.get((metal, ligand), []) if i != ai]
        elif category == "shape_match_diff_chem":
            # similar size + donor profile, but different metal OR ligand class
            cand = [i for i in b.all_idx
                    if (b.metal[i] != metal or b.ligand[i] != ligand)]
            if cand:
                ci = torch.tensor(cand)
                size_d = (b.size[ci] - b.size[ai]).abs()
                donor_d = (b.donor[ci] - b.donor[ai]).abs().sum(-1)
                score = size_d / 10.0 + donor_d           # smaller = more tempting
                # take from the 25 most shape-similar, sampled
                topn = min(25, len(cand))
                near = ci[score.argsort()[:topn]].tolist()
                return self._rand_choice(near)
            pool = []
        else:
            pool = b.all_idx
        out = self._rand_choice(pool)
        return out if out is not None else self._rand_choice(b.all_idx)

    def sample(self, anchor_smis: List[str]) -> torch.LongTensor:
        """Return (B, K) bank indices of hard negatives for each anchor."""
        B = len(anchor_smis)
        out = torch.zeros(B, self.k, dtype=torch.long)
        for bi, smi in enumerate(anchor_smis):
            cat_idx = torch.multinomial(self.cat_p, self.k, replacement=True,
                                        generator=self.g)
            for kk in range(self.k):
                category = self.cats[cat_idx[kk].item()]
                neg = self._sample_one(smi, category)
                if neg is None:
                    neg = self._rand_choice(self.bank.all_idx) or 0
                out[bi, kk] = neg
        return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
    smis = [
        "CC(=O)[O-].CC(=O)[O-].[Pd+2]",
        "Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1",
        "CC(=O)[O-].CC(=O)[O-].[Cu+2]",
        "C=CC[Pd]Cl.C=CC[Pd]Cl",
    ]
    bank = CatalystBank(smis * 5)
    s = HardNegativeSampler(bank, k=4)
    print("metals:", bank.metal[:4], "ligands:", bank.ligand[:4])
    print(s.sample(smis))
