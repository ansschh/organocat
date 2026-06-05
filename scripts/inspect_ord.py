#!/usr/bin/env python3
"""Inspect the parsed ORD TM-catalyzed reactions."""
import pickle, sys, os
from collections import Counter

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.data.ord_parser import TM_SYMBOLS

rxns = pickle.load(open("data/reactions/ord_tm_reactions.pkl", "rb"))
print(f"total TM-catalyzed reactions: {len(rxns)}")
print()

print("=== first 3 reactions ===")
for i, r in enumerate(rxns[:3]):
    rid = (r.get("reaction_id") or "")[:30]
    print(f"--- reaction {i} ({rid}) ---")
    print(f"  reactants: {[x['smiles'][:60] for x in r['reactants'][:3]]}")
    print(f"  catalysts: {[(x['smiles'][:60], x['name'][:40]) for x in r['catalysts'][:2]]}")
    print(f"  products:  {[x['smiles'][:60] for x in r['products'][:2]]}")
print()

metal_counter = Counter()
n_with_explicit_smi = 0
n_with_only_name = 0
n_with_atom_mapping = 0
n_with_both = 0
unique_catalyst_smis = Counter()

for r in rxns:
    for c in r["catalysts"]:
        smi = c.get("smiles", "")
        nm = c.get("name", "")
        if smi:
            n_with_explicit_smi += 1
            unique_catalyst_smis[smi] += 1
        else:
            n_with_only_name += 1
        for tm in TM_SYMBOLS:
            if tm in smi or tm in nm:
                metal_counter[tm] += 1
                break
    has_atom_map = any(":" in x.get("smiles", "") for x in r["reactants"] + r["products"])
    has_smi_both = bool(r["reactants"]) and bool(r["products"])
    if has_atom_map:
        n_with_atom_mapping += 1
    if has_smi_both:
        n_with_both += 1

print(f"reactions with reactants+products SMILES: {n_with_both}")
print(f"reactions with at least one atom-mapped SMILES: {n_with_atom_mapping}")
print(f"catalyst entries with explicit SMILES: {n_with_explicit_smi}")
print(f"catalyst entries with only name (no SMILES): {n_with_only_name}")
print(f"unique catalyst SMILES (top 10):")
for smi, c in unique_catalyst_smis.most_common(10):
    print(f"  {c:>5} x {smi[:70]}")
print()
print("Top 15 metals seen in catalysts:")
for m, c in metal_counter.most_common(15):
    print(f"  {m}: {c}")
