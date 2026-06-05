#!/usr/bin/env python3
"""Parser for tmQM dataset (uiocompcat/tmQM, 2024 release).

Aggregates 4 source files per CSD code:
  tmQM_X[1-3].xyz.gz  coordinates + header metadata (charge, spin, stoich, metal_degree)
  tmQM_X[1-3].BO.gz   Wiberg bond orders (partial bonds included -> hapticity)
  tmQM_X.q            natural atomic partial charges (DFT)
  tmQM_y.csv          DFT properties + SMILES (with dative arrows)
"""
from __future__ import annotations
import csv, gzip, os, re
from collections import defaultdict
from typing import Dict, Optional


TRANSITION_METALS = {
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Ac","Th","U",
}


def _parse_xyz_header(line: str) -> dict:
    out = {}
    for p in [x.strip() for x in line.split("|")]:
        m = re.match(r"([A-Za-z_]+)\s*=\s*(.+)", p)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_xyz_file(path: str) -> Dict[str, dict]:
    out = {}
    with gzip.open(path, "rt") as f:
        while True:
            n_line = f.readline()
            if not n_line:
                break
            n_line = n_line.strip()
            if not n_line:
                continue
            try:
                n = int(n_line)
            except ValueError:
                continue
            header = f.readline().rstrip("\n")
            meta = _parse_xyz_header(header)
            atoms = []
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                t = line.split()
                if len(t) < 4:
                    continue
                atoms.append({
                    "element": t[0], "x": float(t[1]), "y": float(t[2]), "z": float(t[3]),
                    "charge": 0.0,
                })
            csd = meta.get("CSD_code")
            if not csd:
                continue
            try:
                charge = int(meta.get("Charge", "0"))
            except ValueError:
                charge = 0
            try:
                spin = int(meta.get("Spin", "1"))
            except ValueError:
                spin = 1
            try:
                metal_degree = int(meta.get("Metal_q_degree") or meta.get("Metal_node_degree") or "0")
            except ValueError:
                metal_degree = 0
            metal_idx = None
            metal_elem = None
            for i, a in enumerate(atoms):
                if a["element"] in TRANSITION_METALS:
                    metal_idx, metal_elem = i, a["element"]
                    break
            out[csd] = {
                "csd_code": csd, "charge": charge, "spin": spin,
                "stoichiometry": meta.get("Stoichiometry", ""),
                "metal_node_degree": metal_degree,
                "metal_element": metal_elem, "metal_idx": metal_idx,
                "n_atoms": len(atoms), "atoms": atoms, "bonds": [],
            }
    return out


def parse_charge_file(path: str, complexes: Dict[str, dict]) -> None:
    with open(path) as f:
        cur_csd = None
        cur_idx = 0
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("CSD_code"):
                m = re.match(r"CSD_code\s*=\s*(\S+)", line)
                cur_csd = m.group(1) if m else None
                cur_idx = 0
                continue
            t = line.split()
            if len(t) < 2:
                continue
            try:
                q = float(t[1])
            except ValueError:
                continue
            if cur_csd in complexes and cur_idx < len(complexes[cur_csd]["atoms"]):
                complexes[cur_csd]["atoms"][cur_idx]["charge"] = q
                cur_idx += 1


def parse_bo_file(path: str, complexes: Dict[str, dict]) -> None:
    with gzip.open(path, "rt") as f:
        cur_csd = None
        cur_bonds = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("CSD_code"):
                if cur_csd in complexes:
                    complexes[cur_csd]["bonds"] = cur_bonds
                cur_bonds = []
                m = re.match(r"CSD_code\s*=\s*(\S+)", line)
                cur_csd = m.group(1) if m else None
                continue
            t = line.split()
            if len(t) < 3:
                continue
            try:
                atom_idx = int(t[0])
                rest = t[3:]
                k = 0
                while k + 2 < len(rest):
                    try:
                        partner_idx = int(rest[k + 1])
                        bo = float(rest[k + 2])
                        if atom_idx < partner_idx:
                            cur_bonds.append({"i": atom_idx, "j": partner_idx, "order": bo})
                    except ValueError:
                        pass
                    k += 3
            except ValueError:
                continue
        if cur_csd in complexes:
            complexes[cur_csd]["bonds"] = cur_bonds


def parse_properties_file(path: str, complexes: Dict[str, dict]) -> None:
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            csd = row.get("CSD_code")
            if csd not in complexes:
                continue
            def _f(s):
                try:
                    return float(s) if s else None
                except ValueError:
                    return None
            complexes[csd]["smiles"] = row.get("SMILES", "")
            complexes[csd]["properties"] = {
                "electronic_e":   _f(row.get("Electronic_E")),
                "dispersion_e":   _f(row.get("Dispersion_E")),
                "dipole_m":       _f(row.get("Dipole_M")),
                "metal_q":        _f(row.get("Metal_q")),
                "hl_gap":         _f(row.get("HL_Gap")),
                "homo":           _f(row.get("HOMO_Energy")),
                "lumo":           _f(row.get("LUMO_Energy")),
                "polarizability": _f(row.get("Polarizability")),
                "csd_years":      row.get("CSD_years", ""),
            }


def load_tmqm(tmqm_dir: str, limit: Optional[int] = None) -> Dict[str, dict]:
    complexes = {}
    for i in (1, 2, 3):
        p = os.path.join(tmqm_dir, f"tmQM_X{i}.xyz.gz")
        if not os.path.isfile(p):
            print(f"WARN: {p} missing")
            continue
        partial = parse_xyz_file(p)
        complexes.update(partial)
        if limit and len(complexes) >= limit:
            break
    print(f"loaded {len(complexes)} complexes from XYZ files")
    parse_charge_file(os.path.join(tmqm_dir, "tmQM_X.q"), complexes)
    print("charges loaded")
    for i in (1, 2, 3):
        p = os.path.join(tmqm_dir, f"tmQM_X{i}.BO.gz")
        if os.path.isfile(p):
            parse_bo_file(p, complexes)
    print("bond orders loaded")
    parse_properties_file(os.path.join(tmqm_dir, "tmQM_y.csv"), complexes)
    print("properties + SMILES loaded")
    return complexes


def summarize(complexes: Dict[str, dict]) -> None:
    n = len(complexes)
    metals = defaultdict(int)
    charge_dist = defaultdict(int)
    spin_dist = defaultdict(int)
    deg_dist = defaultdict(int)
    no_metal = 0
    no_bonds = 0
    no_smiles = 0
    for c in complexes.values():
        if c["metal_element"] is None:
            no_metal += 1
        else:
            metals[c["metal_element"]] += 1
        charge_dist[c["charge"]] += 1
        spin_dist[c["spin"]] += 1
        deg_dist[c["metal_node_degree"]] += 1
        if not c.get("bonds"):
            no_bonds += 1
        if not c.get("smiles"):
            no_smiles += 1
    print(f"\nTotal complexes: {n}")
    print(f"No metal found: {no_metal}, no bonds: {no_bonds}, no SMILES: {no_smiles}")
    print(f"\nTop 20 metals:")
    for m, c in sorted(metals.items(), key=lambda x: -x[1])[:20]:
        print(f"  {m}: {c}")
    print(f"\nCharge distribution: {dict(sorted(charge_dist.items()))}")
    print(f"Spin distribution: {dict(sorted(spin_dist.items()))}")
    deg_top = sorted(deg_dist.items())[:12]
    print(f"Metal node degree (lowest 12): {dict(deg_top)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmqm-dir", default="data/raw/tmQM/tmQM/tmQM")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cx = load_tmqm(args.tmqm_dir, limit=args.limit)
    summarize(cx)
