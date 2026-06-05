#!/usr/bin/env python3
"""Clean ORD TM reactions + atom-map them with rxnmapper.

Pipeline:
  1. Load raw ORD TM-tagged reactions (~105k).
  2. Filter:
     - Reactant SMILES + product SMILES must parse via rdkit (each).
     - Total reactant heavy atoms in [4, 100], product heavy atoms in [4, 100].
     - At least one catalyst entry has a valid SMILES containing a TM element
       AND has >= 5 heavy atoms (kills bare metal salts + heterogeneous Pd/C).
     - OR a catalyst NAME matches our small known-catalyst lookup (Pd(OAc)2, etc.).
     - Exclude obvious heterogeneous catalysts (Pd carbon / Pd/C, Pt/C, Raney Ni,
       supported catalysts).
     - Drop MFCD codes from catalyst SMILES field.
  3. Atom-map each survivor with rxnmapper. Keep confidence >= 0.5.
  4. Save the clean list to data/reactions/ord_tm_clean.pkl.

This is the dataset for contrastive learning.
"""
from __future__ import annotations
import os, pickle, re, sys, time
from typing import Optional, Tuple, List

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # suppress rdkit parse warnings

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.data.ord_parser import TM_SYMBOLS

# Heterogeneous-catalyst exclusion list (lowercase substring match in name OR SMILES)
HETEROGENEOUS_KEYWORDS = [
    "carbon", "/c", "raney", "graphite", "alumina", "silica", "celite",
    "molecular sieve", "alox", "supported", "mof", "zeolite",
]

# Catalyst SMILES sanity filter — drop MFCD codes (always alphanum, no brackets/dots)
MFCD_PATTERN = re.compile(r"^MFCD\d+$|^[A-Z]{1,4}\d{5,}$")

# Simple known-catalyst name → SMILES (extend as needed)
NAME_TO_SMILES = {
    "palladium(ii) acetate": "CC(=O)O[Pd]OC(C)=O",
    "pd(oac)2": "CC(=O)O[Pd]OC(C)=O",
    "palladium acetate": "CC(=O)O[Pd]OC(C)=O",
    "tetrakis(triphenylphosphine)palladium(0)": "[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    "pd(pph3)4": "[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    "pd(dppf)cl2": "[Pd](Cl)Cl.c1ccc(P(c2ccccc2)C2CCCC2P(c2ccccc2)c2ccccc2)cc1",
    "[(s)-binap-rucl(p-cymene)]cl": "[Cl-].[Cl-].[Ru].",  # placeholder
}


def clean_smi(smi: str) -> Optional[str]:
    """Validate a SMILES string with rdkit and return canonical form, or None."""
    if not smi or len(smi) > 500:
        return None
    if MFCD_PATTERN.match(smi.strip()):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def smiles_has_tm(smi: str) -> bool:
    if not smi:
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return False
    if mol is None:
        return False
    return any(a.GetSymbol() in TM_SYMBOLS for a in mol.GetAtoms())


def is_heterogeneous(name: str, smi: str) -> bool:
    s = (name + " " + smi).lower()
    return any(kw in s for kw in HETEROGENEOUS_KEYWORDS)


def find_homogeneous_catalyst(rxn: dict) -> Optional[Tuple[str, str]]:
    """Return (canonical SMILES, source description) of the first valid
    homogeneous TM catalyst found in this reaction. None if no match."""
    candidates = list(rxn.get("catalysts", [])) + [
        r for r in rxn.get("reactants", []) if r.get("smiles", "")
    ]
    for c in candidates:
        nm = (c.get("name") or "").strip()
        smi = (c.get("smiles") or "").strip()
        if is_heterogeneous(nm, smi):
            continue
        # Try the SMILES first
        canon = clean_smi(smi)
        if canon and smiles_has_tm(canon) and Chem.MolFromSmiles(canon).GetNumAtoms() >= 5:
            return canon, "from_catalyst_smiles"
        # Fall back to name lookup
        nk = nm.lower().strip()
        if nk in NAME_TO_SMILES:
            mapped = NAME_TO_SMILES[nk]
            canon = clean_smi(mapped)
            if canon:
                return canon, "from_name_lookup"
    return None


def clean_reaction(rxn: dict) -> Optional[dict]:
    """Return a clean reaction dict or None if it fails any check."""
    # Reactants
    reactant_smis = []
    for r in rxn.get("reactants", []):
        c = clean_smi(r.get("smiles", ""))
        if c is None:
            continue
        if smiles_has_tm(c):
            continue                       # skip TM-containing reactants (they're catalysts)
        reactant_smis.append(c)
    if not reactant_smis:
        return None
    # Products
    product_smis = []
    for p in rxn.get("products", []):
        c = clean_smi(p.get("smiles", ""))
        if c is None:
            continue
        if smiles_has_tm(c):
            continue
        product_smis.append(c)
    if not product_smis:
        return None
    # Size sanity
    n_react_heavy = sum(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in reactant_smis)
    n_prod_heavy  = sum(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in product_smis)
    if not (4 <= n_react_heavy <= 100 and 4 <= n_prod_heavy <= 100):
        return None
    # Catalyst
    cat = find_homogeneous_catalyst(rxn)
    if cat is None:
        return None
    catalyst_smi, catalyst_source = cat

    return {
        "reaction_id": rxn.get("reaction_id"),
        "reactants_smi": ".".join(reactant_smis),
        "products_smi": ".".join(product_smis),
        "catalyst_smi": catalyst_smi,
        "catalyst_source": catalyst_source,
        "n_react_heavy": n_react_heavy,
        "n_prod_heavy": n_prod_heavy,
        "yield": rxn.get("yield"),
    }


def atom_map_batch(reactions: List[dict], mapper, batch_size: int = 16,
                   confidence_threshold: float = 0.5) -> List[dict]:
    """Run rxnmapper on a list of cleaned reactions. Adds 'mapped_rxn' and
    'mapping_confidence' fields, drops low-confidence entries."""
    out = []
    n = len(reactions)
    for i in range(0, n, batch_size):
        batch = reactions[i:i + batch_size]
        rxn_strs = [f"{r['reactants_smi']}>>{r['products_smi']}" for r in batch]
        try:
            results = mapper.get_attention_guided_atom_maps(rxn_strs)
        except Exception as e:
            # Sometimes individual reactions fail; fall back to one-by-one
            results = []
            for s in rxn_strs:
                try:
                    results.append(mapper.get_attention_guided_atom_maps([s])[0])
                except Exception:
                    results.append({"confidence": 0.0, "mapped_rxn": ""})
        for r, res in zip(batch, results):
            conf = res.get("confidence", 0.0)
            mapped = res.get("mapped_rxn", "")
            if conf < confidence_threshold or not mapped:
                continue
            r2 = dict(r)
            r2["mapped_rxn"] = mapped
            r2["mapping_confidence"] = conf
            out.append(r2)
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= n:
            print(f"  atom-mapped {i + batch_size}/{n}  kept so far: {len(out)}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", default="data/reactions/ord_tm_reactions.pkl")
    ap.add_argument("--out-pkl", default="data/reactions/ord_tm_clean.pkl")
    ap.add_argument("--limit", type=int, default=None, help="testing only")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-mapping", action="store_true", help="just filter; skip atom mapping")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.in_pkl} ...")
    rxns = pickle.load(open(args.in_pkl, "rb"))
    print(f"  loaded {len(rxns)} TM-tagged reactions")
    if args.limit:
        rxns = rxns[:args.limit]

    print(f"\n=== filter pass ===")
    clean = []
    stats = {"no_reactant": 0, "no_product": 0, "size": 0, "no_catalyst": 0, "kept": 0}
    for r in rxns:
        cr = clean_reaction(r)
        if cr is None:
            continue
        clean.append(cr)
    print(f"  kept {len(clean)} / {len(rxns)} ({100*len(clean)/max(1,len(rxns)):.1f}%)")
    print(f"  filter time: {(time.time()-t0):.0f}s")

    if not clean:
        print("nothing to atom-map")
        return

    # Quick stats on catalyst sources
    from collections import Counter
    src = Counter(r["catalyst_source"] for r in clean)
    print(f"  catalyst source breakdown: {dict(src)}")
    cat_counter = Counter(r["catalyst_smi"] for r in clean)
    print(f"  unique catalyst SMILES: {len(cat_counter)}")
    print(f"  top 5 catalysts:")
    for smi, c in cat_counter.most_common(5):
        print(f"     {c:>5}  {smi[:80]}")

    if args.skip_mapping:
        with open(args.out_pkl, "wb") as f:
            pickle.dump(clean, f)
        print(f"\nsaved (no mapping) -> {args.out_pkl}  n={len(clean)}")
        return

    print(f"\n=== atom-mapping ({len(clean)} reactions) ===")
    from rxnmapper import RXNMapper
    mapper = RXNMapper()
    t1 = time.time()
    mapped = atom_map_batch(clean, mapper, batch_size=args.batch_size)
    print(f"  atom-mapping kept {len(mapped)} / {len(clean)} ({100*len(mapped)/max(1,len(clean)):.1f}%, "
          f"confidence >= 0.5).  Time: {(time.time()-t1):.0f}s")

    with open(args.out_pkl, "wb") as f:
        pickle.dump(mapped, f)
    print(f"\nsaved -> {args.out_pkl}  n={len(mapped)}")


if __name__ == "__main__":
    main()
