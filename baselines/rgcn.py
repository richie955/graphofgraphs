"""Flat relational GNN baseline using PyTorch Geometric's RGCNConv."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gog_data.preprocessing import PreprocessedGoG, SCOPES
from gog_model.stage1 import Stage1Initializer


def training_channels(data: PreprocessedGoG, device: torch.device) -> tuple[Tensor, Tensor]:
    """Make forward and reverse channels from Stage 0 training facts only."""
    edges = sorted((edge for scope in SCOPES for edge in data.edge_lists[scope]),
                   key=lambda edge: edge.fact_id)
    if not edges:
        return (torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device))
    source = torch.tensor([edge.source_occurrence_index for edge in edges],
                          dtype=torch.long, device=device)
    target = torch.tensor([edge.target_occurrence_index for edge in edges],
                          dtype=torch.long, device=device)
    relation = torch.tensor([edge.relation_index for edge in edges],
                            dtype=torch.long, device=device)
    relation_count = len(data.relation_id_to_index)
    edge_index = torch.stack((torch.cat((source, target)),
                              torch.cat((target, source))))
    edge_type = torch.cat((relation, relation + relation_count))
    return edge_index, edge_type


class RGCNBaseline(nn.Module):
    """Two R-GCN layers over occurrence nodes, without GoG graph hierarchy."""

    def __init__(self, num_entities: int, num_types: int, num_relations: int,
                 hidden_dim: int = 256, num_bases: int = 8, dropout: float = 0.2):
        super().__init__()
        try:
            from torch_geometric.nn import RGCNConv
        except ImportError as exc:
            raise RuntimeError(
                "R-GCN requires PyTorch Geometric. Install it with: "
                "python -m pip install torch_geometric"
            ) from exc
        self.initializer = Stage1Initializer(num_entities, num_types, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, 2 * num_relations,
                              num_bases=num_bases, root_weight=True)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, 2 * num_relations,
                              num_bases=num_bases, root_weight=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_preprocessed(cls, data: PreprocessedGoG, **kwargs: object) -> "RGCNBaseline":
        return cls(len(data.entities), len(data.type_id_to_index),
                   len(data.relation_id_to_index), **kwargs)

    def forward(self, data: PreprocessedGoG) -> tuple[Tensor, Tensor]:
        device = self.initializer.entity_embedding.weight.device
        types = torch.tensor(data.type_index_by_entity, dtype=torch.long, device=device)
        entities = torch.tensor(data.entity_index_by_occurrence, dtype=torch.long, device=device)
        edge_index, edge_type = training_channels(data, device)
        b_entity, _, z = self.initializer(types, entities)
        z = self.norm1(z + self.dropout(F.relu(self.conv1(z, edge_index, edge_type))))
        z = self.norm2(z + self.dropout(F.relu(self.conv2(z, edge_index, edge_type))))
        return b_entity, z
