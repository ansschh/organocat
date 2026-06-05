#!/usr/bin/env python3
"""Diagnose why the ORD filter rejects everything."""
import pickle, sys
from collections import Counter

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.data.ord_parser import TM_SYMBOLS
from scripts.clean_and_atom_map_ord import (
    clean_smi, smiles_has_tm, is_heterogeneous, find_homogeneous_catalyst,
    NAME_TO_SMILES, HETEROGENEOUS_KEYWORDS,
)

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

rxns = pickle.load(open("data/reactions/ord_tm_reactions.pkl", "rb"))
print(f"total: {len(rxns)}")
N = 5000
sample = rxns[:N]

# Breakdown
no_react = 0; no_prod = 0; size_bad = 0; no_cat = 0; ok = 0
cat_names = Counter()
cat_smiles = Counter()
cat_reasons_rejected = Counter()
for r in sample:
    reactant_smis = []
    for x in r.get("reactants", []):
        c = clean_smi(x.get("smiles", ""))
        if c is None: continue
        if smiles_has_tm(c): continue
        reactant_smis.append(c)
    product_smis = []
    for x in r.get("products", []):
        c = clean_smi(x.get("smiles", ""))
        if c is None: continue
        if smiles_has_tm(c): continue
        product_smis.append(c)
    if not reactant_smis: no_react += 1; continue
    if not product_smis:  no_prod += 1; continue
    if not (4 <= sum(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in reactant_smis) <= 100): size_bad += 1; continue
    cat = find_homogeneous_catalyst(r)
    # Even if no cat, log what was there
    for c in r.get("catalysts", []):
        nm = (c.get("name") or "").strip()
        smi = (c.get("smiles") or "").strip()
        cat_names[nm] += 1
        if smi: cat_smiles[smi] += 1
        # Why rejected?
        if is_heterogeneous(nm, smi): cat_reasons_rejected["heterogeneous"] += 1
        elif not smi: cat_reasons_rejected["no_smiles"] += 1
        else:
            canon = clean_smi(smi)
            if canon is None: cat_reasons_rejected["smi_unparsable_or_mfcd"] += 1
            elif not smiles_has_tm(canon): cat_reasons_rejected["smiles_no_tm"] += 1
            elif Chem.MolFromSmiles(canon).GetNumAtoms() < 5: cat_reasons_rejected["too_small"] += 1
            else: cat_reasons_rejected["should_have_passed_BUG"] += 1
    if cat is None: no_cat += 1; continue
    ok += 1

print(f"\nbreakdown (N={N}):")
print(f"  no_reactant_smi:  {no_react}")
print(f"  no_product_smi:   {no_prod}")
print(f"  size_bad:         {size_bad}")
print(f"  no_homogeneous_catalyst: {no_cat}")
print(f"  kept:             {ok}")
print()
print(f"=== top 15 catalyst names ===")
for nm, c in cat_names.most_common(15):
    print(f"  {c:>5}  {nm[:80]}")
print()
print(f"=== top 15 catalyst SMILES field ===")
for s, c in cat_smiles.most_common(15):
    print(f"  {c:>5}  {s[:80]}")
print()
print(f"=== catalyst rejection reasons ===")
for r, c in cat_reasons_rejected.most_common():
    print(f"  {r}: {c}")
