#!/usr/bin/env python3
"""SE(3)-equivariant encoder for transition-metal complexes.

This is an EGNN-style message-passing network. It uses:

  - Node features: learnable embedding of atomic number Z + scalar features
    (partial charge, is_metal flag).
  - Edge features: euclidean distance + Wiberg bond order.
  - Equivariant coordinate updates: positions are updated by sums of
    (r_j - r_i) weighted by a learned scalar function of (h_i, h_j, |r|, BO).
  - Output: per-graph embedding (mean + metal-only pool) suitable for
    contrastive learning OR regression heads for HOMO/LUMO/etc.

References:
  Satorras, Hoogeboom, Welling. "E(n) Equivariant Graph Neural Networks." ICML 2021.
  arXiv:2102.09844

This is the v0 encoder. NequIP/MACE/etc. swaps can come later if v0
discriminates well.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class EGNNLayer(MessagePassing):
    """One layer of E(n)-equivariant message passing.

    Updates both invariant node features h and equivariant positions x.
    Edge features (distance + BO) are mixed into the message.
    """

    def __init__(self, hidden_dim: int = 128, edge_feat_dim: int = 1,
                 message_dim: int = 64, update_coords: bool = True):
        super().__init__(aggr="add", node_dim=0)
        self.update_coords = update_coords
        self.message_dim = message_dim
        # Edge MLP: (h_i, h_j, |r_ij|^2, edge_attr) -> message vector
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.SiLU(),
        )
        # Node MLP: (h_i, sum_j m_ji) -> h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Coord MLP: message -> scalar weight
        if self.update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(message_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, h, pos, edge_index, edge_attr):
        # Compute pairwise displacement + distance for each edge
        row, col = edge_index   # row=src, col=dst (we use row -> col messages)
        rel = pos[col] - pos[row]                     # (E, 3)
        d2 = (rel * rel).sum(dim=-1, keepdim=True)    # (E, 1)
        # Message construction
        msg_in = torch.cat([h[row], h[col], d2, edge_attr], dim=-1)
        m = self.edge_mlp(msg_in)                      # (E, message_dim)
        # Aggregate messages per node
        m_agg = scatter(m, col, dim=0, dim_size=h.size(0), reduce="add")
        # Update h
        h_new = h + self.node_mlp(torch.cat([h, m_agg], dim=-1))
        # Update coords (equivariant) -- scaled by a learned weight on each message
        if self.update_coords:
            w = self.coord_mlp(m)                      # (E, 1)
            coord_contrib = rel * w                    # (E, 3), equivariant
            # avoid blowing up: normalize by distance + 1
            coord_contrib = coord_contrib / (d2.sqrt() + 1.0)
            pos_delta = scatter(coord_contrib, col, dim=0, dim_size=pos.size(0), reduce="add")
            pos_new = pos + pos_delta
        else:
            pos_new = pos
        return h_new, pos_new


class ComplexEGNN(nn.Module):
    """End-to-end encoder for TM complexes.

    Inputs (per graph):
      data.pos        (N, 3)  positions
      data.z          (N,)    atomic numbers
      data.charges    (N,)    partial charges
      data.metal_mask (N,)    bool
      data.edge_index (2, E)  bonds
      data.edge_attr  (E, 1)  bond order
      data.batch      (N,)    graph assignment (PyG batch)

    Output:
      embedding (B, embed_dim) where B = number of graphs in batch
    """

    def __init__(self, num_z: int = 100, hidden_dim: int = 128,
                 n_layers: int = 4, embed_dim: int = 128,
                 update_coords: bool = True):
        super().__init__()
        self.z_embed = nn.Embedding(num_z + 1, hidden_dim)
        # Per-atom scalar feature projection: partial charge + is_metal
        self.scalar_proj = nn.Linear(2, hidden_dim)
        self.input_mix = nn.Linear(2 * hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, edge_feat_dim=1,
                      message_dim=hidden_dim // 2, update_coords=update_coords)
            for _ in range(n_layers)
        ])

        # Pool head: combine mean-of-all + metal-only signal
        self.pool_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        # Initial node features
        h_z = self.z_embed(data.z)                     # (N, H)
        s = torch.stack([data.charges, data.metal_mask.float()], dim=-1)
        h_s = self.scalar_proj(s)                       # (N, H)
        h = self.input_mix(torch.cat([h_z, h_s], dim=-1))  # (N, H)
        pos = data.pos

        for layer in self.layers:
            h, pos = layer(h, pos, data.edge_index, data.edge_attr)

        # Graph-level pooling: mean of all nodes + metal node
        batch = data.batch if data.batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_mean = scatter(h, batch, dim=0, reduce="mean")               # (B, H)
        # metal-only pool: exactly one metal per graph, selected via the boolean
        # metal_mask (concatenates correctly under PyG batching; no index offset).
        h_metal = h[data.metal_mask]                                   # (B, H)  -- one per graph
        graph_feat = torch.cat([h_mean, h_metal], dim=-1)
        emb = self.pool_head(graph_feat)
        return emb


class RegressionHead(nn.Module):
    """Simple regression head for pretraining (predict HOMO/LUMO/gap from complex emb)."""

    def __init__(self, embed_dim: int = 128, n_targets: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, n_targets),
        )

    def forward(self, emb):
        return self.net(emb)
