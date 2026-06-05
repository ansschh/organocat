#!/usr/bin/env python3
"""PyG Dataset for tmQM complexes.

Loads from the pickle written by scripts/build_tmqm_index.py and yields
torch_geometric.data.Data objects with:

  pos        (N, 3) float  Cartesian coordinates (xTB-optimized)
  z          (N,) long     atomic numbers
  charges    (N,) float    natural atomic partial charges (DFT)
  edge_index (2, E) long   bond connectivity (undirected; both i->j and j->i)
  edge_attr  (E, 1) float  Wiberg bond order
  metal_mask (N,) bool     True at the metal atom
  metal_idx  long          index of the metal atom (single int per graph)
  csd_code   str           CSD code (for retrieval lookups)

Targets (y) available per complex:
  electronic_e, dispersion_e, dipole_m, metal_q, hl_gap, homo, lumo, polarizability
"""
from __future__ import annotations
import os, pickle
from typing import List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset


ELEMENT_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
    "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88,
    "Ac": 89, "Th": 90, "U": 92,
}


def complex_to_pyg(c: dict, target_keys: Optional[List[str]] = None) -> Optional[Data]:
    """Convert a parsed tmQM complex dict into a torch_geometric Data object.

    Returns None if the complex is malformed (no metal, no bonds, etc.).
    """
    if c.get("metal_idx") is None or not c.get("bonds"):
        return None
    atoms = c["atoms"]
    n = len(atoms)
    if n < 5:
        return None

    pos = torch.tensor([[a["x"], a["y"], a["z"]] for a in atoms], dtype=torch.float)
    z = torch.tensor([ELEMENT_TO_Z.get(a["element"], 0) for a in atoms], dtype=torch.long)
    charges = torch.tensor([a["charge"] for a in atoms], dtype=torch.float)

    # Bonds are 1-indexed in tmQM; convert to 0-indexed. Emit both directions for PyG.
    src, dst, bo = [], [], []
    for b in c["bonds"]:
        i = b["i"] - 1
        j = b["j"] - 1
        if i < 0 or j < 0 or i >= n or j >= n:
            continue
        # Threshold extremely weak (<0.05) BOs as noise; keep partial bonds (hapticity)
        order = b["order"]
        if order < 0.05:
            continue
        src += [i, j]
        dst += [j, i]
        bo += [order, order]
    if not src:
        return None
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(bo, dtype=torch.float).unsqueeze(-1)

    metal_idx = c["metal_idx"]
    metal_mask = torch.zeros(n, dtype=torch.bool)
    metal_mask[metal_idx] = True

    data = Data(pos=pos, z=z, charges=charges,
                edge_index=edge_index, edge_attr=edge_attr,
                metal_mask=metal_mask)
    data.metal_idx = torch.tensor([metal_idx], dtype=torch.long)
    data.metal_z = torch.tensor([z[metal_idx].item()], dtype=torch.long)
    data.csd_code = c["csd_code"]
    data.total_charge = torch.tensor([c.get("charge", 0)], dtype=torch.float)
    data.spin = torch.tensor([c.get("spin", 0)], dtype=torch.float)
    data.coord_number = torch.tensor([c.get("metal_node_degree", 0)], dtype=torch.float)

    # Targets
    props = c.get("properties") or {}
    target_keys = target_keys or ["hl_gap", "homo", "lumo", "metal_q", "dipole_m", "polarizability"]
    target_vals = []
    for k in target_keys:
        v = props.get(k)
        target_vals.append(float(v) if v is not None else float("nan"))
    data.y = torch.tensor(target_vals, dtype=torch.float).unsqueeze(0)  # (1, T)
    data.target_keys = target_keys
    return data


class TMQMDataset(InMemoryDataset):
    """In-memory dataset built from the parsed tmQM pickle."""

    def __init__(self, pickle_path: str, target_keys: Optional[List[str]] = None,
                 metals: Optional[List[str]] = None, max_atoms: Optional[int] = None):
        """
        Args:
          pickle_path:  path to data/tmqm/tmqm_full.pkl
          target_keys:  which properties to include in y
          metals:       restrict to these metal elements (e.g. ["Ir", "Rh", "Ru"])
          max_atoms:    drop complexes larger than this (memory control)
        """
        # we don't call super().__init__ — build data manually
        with open(pickle_path, "rb") as f:
            complexes = pickle.load(f)
        self.target_keys = target_keys or ["hl_gap", "homo", "lumo", "metal_q", "dipole_m", "polarizability"]
        kept = []
        skipped_no_metal = 0; skipped_too_big = 0; skipped_metal_filter = 0
        for csd, c in complexes.items():
            if metals and c.get("metal_element") not in metals:
                skipped_metal_filter += 1; continue
            if max_atoms and c.get("n_atoms", 0) > max_atoms:
                skipped_too_big += 1; continue
            d = complex_to_pyg(c, target_keys=self.target_keys)
            if d is None:
                skipped_no_metal += 1; continue
            kept.append(d)
        self._data_list = kept
        self.csd_to_idx = {d.csd_code: i for i, d in enumerate(kept)}
        print(f"TMQMDataset: kept {len(kept)} / {len(complexes)}  "
              f"(skipped no_metal={skipped_no_metal}, too_big={skipped_too_big}, "
              f"metal_filter={skipped_metal_filter})")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

    def get_by_csd(self, csd_code: str):
        return self._data_list[self.csd_to_idx[csd_code]] if csd_code in self.csd_to_idx else None

    def metals_distribution(self):
        from collections import Counter
        return Counter(d.csd_code and (lambda: ELEMENT_TO_Z and None) for d in self._data_list)


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="data/tmqm/tmqm_full.pkl")
    ap.add_argument("--metals", nargs="*", default=None,
                    help="filter to these metals, e.g. Ir Rh Ru Pd Pt")
    ap.add_argument("--max-atoms", type=int, default=200)
    args = ap.parse_args()
    ds = TMQMDataset(args.pickle, metals=args.metals, max_atoms=args.max_atoms)
    print(f"\nFirst sample:")
    d = ds[0]
    print(d)
    print(f"  csd_code={d.csd_code}  metal_z={d.metal_z.item()}  total_charge={d.total_charge.item()}")
    print(f"  y (targets {d.target_keys}): {d.y}")
