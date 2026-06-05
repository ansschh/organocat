#!/usr/bin/env python3
"""Reaction-graph encoder.

Inputs are atom-mapped reaction graphs: a reactant graph + a product graph +
the bond-change set (delta_bonds). Output is a fixed-dim reaction embedding.

Encoding strategy (v0, message-passing GNN over the union graph with bond-change
edge attributes):

  1. Build a single graph with all reactant atoms (atom-mapped). Each atom node
     carries its element, formal charge, aromaticity, hybridization, degree.
  2. Edges: include all reactant bonds AND all product bonds. Each edge has:
       - bond_order_reactant
       - bond_order_product
       - delta = bond_order_product - bond_order_reactant
       - reactant_only / product_only / persists flags
  3. Several rounds of edge-aware message passing.
  4. Pool over atoms; project to embed_dim.

The delta-bond representation is the FlowER-flavored signal: edges where
delta != 0 are the reaction centers. That's where the chemistry happens.

For atoms without explicit atom-mapping, we fall back to a default mapping
that matches by canonical SMILES position; better atom-mapping (RDKit's
AtomMapNumber via reaction templates) is left as a TODO.
"""
from __future__ import annotations
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from rdkit import Chem
from rdkit.Chem import AllChem


# Atom features (compact set; can extend with E3FP/ECFP if needed)
ATOMIC_NUMS = list(range(1, 95))           # H .. Pu
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP,
                  Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2,
                  Chem.rdchem.HybridizationType.S,
                  Chem.rdchem.HybridizationType.UNSPECIFIED]
HYBRID_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATIONS)}


def atom_features(atom: Chem.Atom) -> List[float]:
    """Build atom feature vector (length ~30)."""
    z = atom.GetAtomicNum()
    z_one_hot = [0.0] * 95
    if 0 < z < 95:
        z_one_hot[z] = 1.0
    return z_one_hot + [
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
        float(atom.GetDegree()),
        float(HYBRID_TO_IDX.get(atom.GetHybridization(), len(HYBRIDIZATIONS))),
    ]


ATOM_FEAT_DIM = 100   # 95 z one-hot + 5 scalars


def parse_reaction_smiles(rxn_smiles: str) -> Optional[Data]:
    """Parse an atom-mapped reaction SMILES into a PyG Data object with
    delta-bond edge attributes.

    Args:
      rxn_smiles: "react1.react2.>>prod1.prod2." with atom-map numbers like [CH4:1]
    Returns:
      PyG Data with .x (atom features), .edge_index, .edge_attr (BO_r, BO_p, delta,
      is_reactant_only, is_product_only, is_reaction_center).
      None if parsing fails.
    """
    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        return None
    reactants_smi, products_smi = parts
    react = Chem.MolFromSmiles(reactants_smi.replace(">>", ""))
    prod  = Chem.MolFromSmiles(products_smi)
    if react is None or prod is None:
        return None

    # Map atom-map-number -> atom in each side
    r_amn = {a.GetAtomMapNum(): a for a in react.GetAtoms() if a.GetAtomMapNum() > 0}
    p_amn = {a.GetAtomMapNum(): a for a in prod.GetAtoms()  if a.GetAtomMapNum() > 0}
    common = sorted(set(r_amn) & set(p_amn))
    if not common:
        # No atom-mapping — fall back to just-reactant graph
        common = []
        for i, a in enumerate(react.GetAtoms()):
            a.SetAtomMapNum(i + 1)
            common.append(i + 1)
        # Product atoms not mapped — for v0 we ignore them
        r_amn = {a.GetAtomMapNum(): a for a in react.GetAtoms()}
        p_amn = {}

    # Build node feature matrix indexed by common atom-map order
    amn_to_idx = {amn: i for i, amn in enumerate(common)}
    n_nodes = len(common)
    x = torch.zeros(n_nodes, ATOM_FEAT_DIM)
    for amn, i in amn_to_idx.items():
        x[i] = torch.tensor(atom_features(r_amn[amn]))

    # Reactant bonds + product bonds (between common atoms)
    def get_bond_orders(mol, amn_to_idx):
        out = {}
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
            if a1 in amn_to_idx and a2 in amn_to_idx:
                i, j = amn_to_idx[a1], amn_to_idx[a2]
                key = (min(i, j), max(i, j))
                out[key] = float(bond.GetBondTypeAsDouble())
        return out

    r_bonds = get_bond_orders(react, amn_to_idx)
    p_bonds = get_bond_orders(prod, amn_to_idx)
    all_keys = set(r_bonds) | set(p_bonds)
    if not all_keys:
        return None

    src, dst, attr = [], [], []
    for (i, j) in all_keys:
        bo_r = r_bonds.get((i, j), 0.0)
        bo_p = p_bonds.get((i, j), 0.0)
        delta = bo_p - bo_r
        feat = [bo_r, bo_p, delta,
                float(bo_r > 0 and bo_p == 0),    # reactant_only
                float(bo_p > 0 and bo_r == 0),    # product_only
                float(abs(delta) > 0.01)]          # is_reaction_center
        src += [i, j]; dst += [j, i]
        attr += [feat, feat]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.n_nodes = n_nodes
    return data


class ReactionGNN(nn.Module):
    """Message-passing GNN over the reaction graph (union of reactant + product
    bonds, edge-annotated with bond change delta).
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4,
                 embed_dim: int = 128, edge_feat_dim: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(ATOM_FEAT_DIM, hidden_dim)
        self.layers = nn.ModuleList([
            ReactionEdgeMP(hidden_dim, edge_feat_dim) for _ in range(n_layers)
        ])
        self.pool_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, data):
        h = self.input_proj(data.x)
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
        # Pool: mean over all atoms + mean over reaction-center atoms
        batch = data.batch if data.batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_mean = scatter(h, batch, dim=0, reduce="mean")

        # Reaction-center atoms: atoms that touch any reaction-center edge
        # edge_attr column 5 = is_reaction_center (per edge)
        ec = data.edge_attr[:, 5] > 0.5
        ec_atoms = data.edge_index[0][ec]
        # weight atoms by how often they appear at reaction-center edges
        if ec_atoms.numel() > 0:
            counts = torch.zeros(h.size(0), device=h.device)
            counts.scatter_add_(0, ec_atoms, torch.ones_like(ec_atoms, dtype=torch.float))
            counts = counts.unsqueeze(-1)
            h_center = scatter(h * counts, batch, dim=0, reduce="sum") / \
                       (scatter(counts, batch, dim=0, reduce="sum").clamp(min=1.0))
        else:
            h_center = h_mean

        graph_feat = torch.cat([h_mean, h_center], dim=-1)
        return self.pool_head(graph_feat)


class ReactionEdgeMP(MessagePassing):
    """Message passing with edge features mixed into the message."""

    def __init__(self, hidden_dim: int, edge_feat_dim: int):
        super().__init__(aggr="add", node_dim=0)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        msg_in = torch.cat([h[row], h[col], edge_attr], dim=-1)
        m = self.msg_mlp(msg_in)
        m_agg = scatter(m, col, dim=0, dim_size=h.size(0), reduce="add")
        return h + self.upd_mlp(torch.cat([h, m_agg], dim=-1))


if __name__ == "__main__":
    # Quick test
    rxn = "[CH3:1][CH2:2][OH:3].[Cl:4][H:5]>>[CH3:1][CH2:2][Cl:4].[OH:3][H:5]"
    data = parse_reaction_smiles(rxn)
    print(data)
    model = ReactionGNN(embed_dim=64)
    emb = model(data)
    print(f"reaction embedding: {emb.shape}  norm={emb.norm():.3f}")
