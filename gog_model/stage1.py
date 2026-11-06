"""Stage 1: create one starting vector per entity and copy it to its graph nodes."""

import torch
from torch import Tensor, nn


class Stage1Initializer(nn.Module):
    """Canonical entity embeddings plus shared occurrence normalization."""

    def __init__(self, num_entities: int, num_types: int, hidden_dim: int = 256):
        super().__init__()
        if num_entities < 0 or num_types < 1 or hidden_dim < 1:
            raise ValueError("num_entities must be nonnegative; num_types and hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.occurrence_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.entity_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

    def forward(self, type_index_by_entity: Tensor, entity_index_by_occurrence: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if type_index_by_entity.ndim != 1:
            raise ValueError("type_index_by_entity must be rank 1")
        if entity_index_by_occurrence.ndim != 1:
            raise ValueError("entity_index_by_occurrence must be rank 1")
        if type_index_by_entity.numel() != self.entity_embedding.num_embeddings:
            raise ValueError("type_index_by_entity length must match entity embedding table")
        if type_index_by_entity.device != self.entity_embedding.weight.device:
            type_index_by_entity = type_index_by_entity.to(self.entity_embedding.weight.device)
        occurrence_map = entity_index_by_occurrence.to(self.entity_embedding.weight.device)
        b_entity = self.entity_embedding.weight + self.type_embedding(type_index_by_entity.long())
        occurrence_base = b_entity.index_select(0, occurrence_map.long())
        delta_occurrence = torch.zeros_like(occurrence_base)
        z_occurrence = self.occurrence_norm(occurrence_base + delta_occurrence)
        return b_entity, delta_occurrence, z_occurrence
