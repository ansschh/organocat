#!/usr/bin/env python3
"""Cheap chemical descriptors for a catalyst SMILES.

Shared infrastructure for:
  - leave-metal-out / leave-ligand-class-out splits
  - hard-negative sampling (same metal / wrong ligand, etc.)
  - the metal-frequency-prior baseline

Everything here is rdkit-only and deterministic. We deliberately keep it
robust to the messy ionic / disconnected catalyst SMILES that ORD produces
(e.g. "CC(=O)[O-].CC(=O)[O-].[Pd+2]", where the metal is not bonded to
anything). When the metal has explicit neighbors we use them; otherwise we
fall back to a whole-molecule donor-atom scan.
"""
from __future__ import annotations
import sys
from functools import lru_cache
from typing import Dict, Optional

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))
from src.data.ord_parser import TM_SYMBOLS

HALIDES = {"F", "Cl", "Br", "I"}

# Ligand-class buckets, in priority order (first match wins).
LIGAND_CLASSES = ["Cp_arene", "NHC", "phosphine", "N_donor", "O_donor",
                  "halide_only", "other"]


def _has_cp_or_arene(mol: Chem.Mol, metal_idx: Optional[int]) -> bool:
    """Cyclopentadienyl / arene coordination: an all-carbon ring (5 or 6) that
    is aromatic or anionic-cyclopentadienyl. Approximate: any all-carbon
    aromatic 5- or 6-ring present in the molecule."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetSymbol() == "C" for a in atoms):
            # aromatic ring or a cyclopentadienyl carbanion
            if all(a.GetIsAromatic() for a in atoms):
                return True
            if any(a.GetFormalCharge() < 0 for a in atoms):
                return True
    return False


def _has_nhc(mol: Chem.Mol) -> bool:
    """N-heterocyclic carbene: a carbene-like carbon flanked by two nitrogens
    (imidazol-2-ylidene motif). Approximate via SMARTS on a divalent C with two
    aromatic-N neighbors in a 5-ring."""
    patt = Chem.MolFromSmarts("[#6;X2,X1]([#7])[#7]")
    if patt is None:
        return False
    return mol.HasSubstructMatch(patt)


@lru_cache(maxsize=20000)
def catalyst_descriptors(smi: str) -> Optional[Dict]:
    """Return {metal, ligand_class, has_P, has_N, has_O, n_halide,
    n_atoms, charge} for a catalyst SMILES, or None if unparsable / no metal."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    metals = [a for a in mol.GetAtoms() if a.GetSymbol() in TM_SYMBOLS]
    if not metals:
        return None
    metal_atom = metals[0]
    metal = metal_atom.GetSymbol()
    metal_idx = metal_atom.GetIdx()

    # Donor atoms: prefer explicit metal neighbors; else whole-molecule scan.
    neigh = list(metal_atom.GetNeighbors())
    if neigh:
        donor_syms = [n.GetSymbol() for n in neigh]
    else:
        donor_syms = [a.GetSymbol() for a in mol.GetAtoms()
                      if a.GetSymbol() not in TM_SYMBOLS]

    has_P = "P" in donor_syms
    has_N = "N" in donor_syms
    has_O = "O" in donor_syms
    n_halide = sum(1 for s in donor_syms if s in HALIDES)

    # Ligand-class bucket (priority order).
    if _has_cp_or_arene(mol, metal_idx):
        lclass = "Cp_arene"
    elif _has_nhc(mol):
        lclass = "NHC"
    elif has_P:
        lclass = "phosphine"
    elif has_N:
        lclass = "N_donor"
    elif has_O:
        lclass = "O_donor"
    elif n_halide > 0 and not (has_P or has_N or has_O):
        lclass = "halide_only"
    else:
        lclass = "other"

    charge = Chem.GetFormalCharge(mol)
    return {
        "metal": metal, "ligand_class": lclass,
        "has_P": has_P, "has_N": has_N, "has_O": has_O,
        "n_halide": n_halide, "n_atoms": mol.GetNumAtoms(), "charge": charge,
    }


if __name__ == "__main__":
    tests = [
        "c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2))cc1",
        "CC(=O)[O-].CC(=O)[O-].[Pd+2]",
        "C=CC[Pd]Cl.C=CC[Pd]Cl",
        "CC(=O)[O-].CC(=O)[O-].[Cu+2]",
        "Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1",
    ]
    for s in tests:
        print(f"{s[:55]:55s} -> {catalyst_descriptors(s)}")
