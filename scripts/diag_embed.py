#!/usr/bin/env python3
"""Diagnose catalyst 3D-embedding failures (the '0 pairs' bug).

Prints, for the first few real catalyst SMILES: parse status, atom count, the
ETKDG embed return code, and any exception — plus a direct call to the actual
embed_catalyst_3d so its real traceback (if any) surfaces.
"""
import sys, pickle, traceback
sys.path.insert(0, __import__("os").environ.get("ORGANOCAT_ROOT", "."))

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

print("=== rdkit version ===")
import rdkit
print(rdkit.__version__)
print("AllChem.ETKDGv3 present:",
      hasattr(__import__("rdkit.Chem.AllChem", fromlist=["x"]), "ETKDGv3"))
try:
    from rdkit.Chem import rdDistGeom
    print("rdDistGeom.ETKDGv3 present:", hasattr(rdDistGeom, "ETKDGv3"))
except Exception as e:
    print("rdDistGeom import failed:", repr(e))

clean = pickle.load(open("data/reactions/ord_tm_clean.pkl", "rb"))
print(f"\nclean reactions: {len(clean)}")
cats = list({r["catalyst_smi"] for r in clean})
print(f"unique catalysts: {len(cats)}")

from rdkit.Chem import rdDistGeom as DG
for smi in cats[:6]:
    print("=" * 60)
    print("SMI:", smi[:80])
    m = Chem.MolFromSmiles(smi)
    print(" MolFromSmiles:", m is not None, "| heavy:", m.GetNumAtoms() if m else None)
    if m is None:
        continue
    try:
        mh = Chem.AddHs(m)
        print(" with Hs:", mh.GetNumAtoms())
        p = DG.ETKDGv3(); p.randomSeed = 42; p.useRandomCoords = True
        code = DG.EmbedMolecule(mh, p)
        print(" embed code:", code, "(0 = success)")
    except Exception as e:
        print(" EXCEPTION:", repr(e))

print("=" * 60)
print("DIRECT embed_catalyst_3d on first 3 catalysts (post-fix):")
from src.data.contrastive_dataset import embed_catalyst_3d
for smi in cats[:3]:
    try:
        r = embed_catalyst_3d(smi)
        print(f"  {smi[:50]:50s} -> {'OK' if r else 'None'}")
    except Exception:
        traceback.print_exc()
