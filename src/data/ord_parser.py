#!/usr/bin/env python3
"""Parse Open Reaction Database (ORD) files -> list of reaction dicts, tagging
transition-metal-catalyzed reactions.

RECONSTRUCTED (the pod-built original is gone). Robust to ORD's two on-disk
formats:
  * .pb.gz  : gzipped serialized ord_schema Dataset protobuf
  * .parquet: a table with one column holding serialized Reaction protobuf
              bytes (column name varies across exports, so we auto-detect it).

Each output dict:
  {
    "reaction_id": str,
    "reactants":  [{"smiles": str, "name": str}, ...],   # role REACTANT
    "reagents":   [...],                                  # role REAGENT
    "catalysts":  [{"smiles": str, "name": str}, ...],    # role CATALYST
    "products":   [{"smiles": str, "name": str}, ...],
    "yield": float | None,
  }

We tag a reaction as TM-catalyzed if any catalyst/reagent component contains a
transition metal (by rdkit atom scan on SMILES, or a metal-name match). The
*strict* homogeneous filter happens downstream in clean_and_atom_map_ord.py;
here we stay permissive.
"""
from __future__ import annotations
import glob
import gzip
import os
import re
import sys
from typing import Dict, Iterator, List, Optional

# ---- transition-metal vocabulary -------------------------------------------

TM_SYMBOLS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

# Metal element names (lowercase) for matching free-text catalyst names.
TM_NAMES = {
    "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
    "cobalt", "nickel", "copper", "zinc", "yttrium", "zirconium", "niobium",
    "molybdenum", "technetium", "ruthenium", "rhodium", "palladium", "silver",
    "cadmium", "lanthanum", "hafnium", "tantalum", "tungsten", "rhenium",
    "osmium", "iridium", "platinum", "gold", "mercury",
    # common abbreviations / adjectival forms
    "pd", "pt", "ru", "rh", "ir", "ni", "cu", "fe", "co", "mn", "pallad",
}

# Two-letter symbols must be tried before one-letter to avoid mis-tokenizing.
_TM_SORTED = sorted(TM_SYMBOLS, key=len, reverse=True)
_TM_REGEX = re.compile(r"\[[^\]]*?(" + "|".join(_TM_SORTED) + r")[^\]]*?\]")

# Lazy rdkit (optional but preferred for SMILES TM detection).
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False


def smiles_has_tm(smi: str) -> bool:
    if not smi:
        return False
    if _HAVE_RDKIT:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return any(a.GetSymbol() in TM_SYMBOLS for a in mol.GetAtoms())
        except Exception:
            pass
    return bool(_TM_REGEX.search(smi))


def name_has_tm(name: str) -> bool:
    if not name:
        return False
    n = name.lower()
    return any(tok in n for tok in TM_NAMES)


# ---- ord_schema protobuf access --------------------------------------------

def _load_proto_modules():
    """Import ord_schema reaction protobuf; raise a clear error if missing."""
    try:
        from ord_schema.proto import reaction_pb2  # noqa
        from ord_schema.proto import dataset_pb2   # noqa
        return reaction_pb2, dataset_pb2
    except Exception as e:
        raise RuntimeError(
            "ord_schema not installed. `pip install ord-schema` "
            f"(import error: {e})")


# CompoundIdentifier.IdentifierType:  SMILES=2, NAME=6  (stable ORD enum)
_ID_SMILES = 2
_ID_NAME = 6
# ReactionRole: REACTANT=1, REAGENT=2, SOLVENT=3, CATALYST=4
_ROLE_REACTANT, _ROLE_REAGENT, _ROLE_SOLVENT, _ROLE_CATALYST = 1, 2, 3, 4


def _compound_to_dict(cmp) -> Dict[str, str]:
    smi = next((i.value for i in cmp.identifiers if i.type == _ID_SMILES), "")
    name = next((i.value for i in cmp.identifiers if i.type == _ID_NAME), "")
    return {"smiles": smi, "name": name}


def _extract_yield(rxn) -> Optional[float]:
    for outcome in rxn.outcomes:
        for prod in outcome.products:
            for m in prod.measurements:
                # ProductMeasurement.MeasurementType.YIELD == 3 (ORD enum)
                if m.type == 3 and m.HasField("percentage"):
                    return float(m.percentage.value)
    return None


def reaction_proto_to_dict(rxn) -> Dict:
    reactants, reagents, catalysts = [], [], []
    for _key, rinput in rxn.inputs.items():
        for cmp in rinput.components:
            d = _compound_to_dict(cmp)
            role = cmp.reaction_role
            if role == _ROLE_CATALYST:
                catalysts.append(d)
            elif role == _ROLE_REAGENT or role == _ROLE_SOLVENT:
                reagents.append(d)
            else:                       # REACTANT or unspecified
                reactants.append(d)
    products = []
    for outcome in rxn.outcomes:
        for prod in outcome.products:
            products.append(_compound_to_dict(prod))
    return {
        "reaction_id": rxn.reaction_id,
        "reactants": reactants,
        "reagents": reagents,
        "catalysts": catalysts,
        "products": products,
        "yield": _extract_yield(rxn),
    }


def is_tm_catalyzed(rdict: Dict) -> bool:
    """Permissive TM tag: any catalyst/reagent component carries a TM."""
    for grp in ("catalysts", "reagents"):
        for c in rdict.get(grp, []):
            if smiles_has_tm(c.get("smiles", "")) or name_has_tm(c.get("name", "")):
                return True
    # some ORD datasets file the metal as a reactant
    for c in rdict.get("reactants", []):
        if smiles_has_tm(c.get("smiles", "")):
            return True
    return False


# ---- file iteration ---------------------------------------------------------

def _detect_proto_column(df, reaction_pb2) -> Optional[str]:
    """Find the parquet column holding serialized Reaction protobufs by trying
    FromString on the first non-null value of each candidate column."""
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        val = series.iloc[0]
        if not isinstance(val, (bytes, bytearray)):
            continue
        try:
            reaction_pb2.Reaction.FromString(bytes(val))
            return col
        except Exception:
            continue
    return None


def iter_reactions_in_file(path: str) -> Iterator:
    """Yield Reaction protobuf messages from a .pb.gz or .parquet ORD file."""
    reaction_pb2, dataset_pb2 = _load_proto_modules()
    if path.endswith(".pb.gz") or path.endswith(".pb"):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            data = f.read()
        ds = dataset_pb2.Dataset.FromString(data)
        for rxn in ds.reactions:
            yield rxn
    elif path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        col = _detect_proto_column(df, reaction_pb2)
        if col is None:
            raise RuntimeError(
                f"could not auto-detect serialized-protobuf column in {path}; "
                f"columns={list(df.columns)}. Run parse_ord.py --inspect.")
        for val in df[col].dropna():
            try:
                yield reaction_pb2.Reaction.FromString(bytes(val))
            except Exception:
                continue
    else:
        raise ValueError(f"unsupported ORD file type: {path}")


def _dataset_id(path: str) -> str:
    b = os.path.basename(path)
    for ext in (".pb.gz", ".parquet", ".pb"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return b


def find_ord_files(root: str) -> List[str]:
    """Return one file per ORD dataset. Datasets may ship as BOTH .pb.gz and
    .parquet (and some as only one). Dedupe by dataset id, preferring the
    canonical .pb.gz so every dataset is parsed exactly once."""
    all_files = (glob.glob(os.path.join(root, "**", "*.pb.gz"), recursive=True)
                 + glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    by_id: dict = {}
    for f in all_files:
        ds = _dataset_id(f)
        # prefer .pb.gz; otherwise keep whatever we have
        if ds not in by_id or f.endswith(".pb.gz"):
            by_id[ds] = f
    return sorted(by_id.values())


def parse_ord_root(root: str, tm_only: bool = True,
                   limit_files: Optional[int] = None,
                   progress_every: int = 20) -> List[Dict]:
    files = find_ord_files(root)
    if limit_files:
        files = files[:limit_files]
    print(f"[ord_parser] {len(files)} ORD files under {root}", flush=True)
    out: List[Dict] = []
    n_total = 0
    for fi, path in enumerate(files):
        try:
            for rxn in iter_reactions_in_file(path):
                n_total += 1
                d = reaction_proto_to_dict(rxn)
                if (not tm_only) or is_tm_catalyzed(d):
                    out.append(d)
        except Exception as e:
            print(f"  [warn] {os.path.basename(path)}: {e}", flush=True)
        if (fi + 1) % progress_every == 0 or fi == len(files) - 1:
            print(f"  file {fi+1}/{len(files)}  scanned={n_total}  kept={len(out)}",
                  flush=True)
    return out


if __name__ == "__main__":
    # smoke test for TM detection
    tests = [("CC(=O)O[Pd]OC(C)=O", ""), ("CCO", ""), ("", "Palladium acetate")]
    for smi, nm in tests:
        print(f"{smi!r:30s} {nm!r:25s} -> tm={smiles_has_tm(smi) or name_has_tm(nm)}")
