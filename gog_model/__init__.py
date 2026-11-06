"""Simple public entry point for the GoG model stages."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from gog_preprocessing import PreprocessedGoG
from .stage1 import Stage1Initializer
from .stage2 import Stage2LocalCompGCN
from .stage3 import Stage3CrossGraphAttention
from .stage4 import Stage4GraphReadout
from .stage5 import Stage5GraphEntityFusion
from .stage6 import Stage6RootCompGCN
from .stage7 import Stage7IdentitySync
from .stage8 import Stage8ContextFeedback
from .stage9 import ComplExDecoder


@dataclass
class EncoderSliceOutput:
    b_entity: Tensor
    delta_occurrence: Tensor
    z_occurrence: Tensor
    z_local: Tensor
    z_cross: Tensor
    local_channel_count: int
    cross_channel_count: int


@dataclass
class EncoderRoundOutput:
    z_local: Tensor
    z_cross: Tensor
    h_graph: Tensor
    z_owner: Tensor
    z_root: Tensor
    b_entity: Tensor
    z_next: Tensor
    local_channel_count: int
    cross_channel_count: int
    root_channel_count: int


@dataclass
class GoGEncoderOutput:
    b_entity: Tensor
    z_occurrence: Tensor
    h_graph: Tensor
    rounds: list[EncoderRoundOutput]


class GoGEncoderStages1To3(nn.Module):
    """Integrated Stage 1→2→3 encoder slice for a Stage 0 object."""

    def __init__(self, num_entities: int, num_types: int, num_relations: int,
                 hidden_dim: int = 256, num_bases: int = 8, num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.initializer = Stage1Initializer(num_entities, num_types, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)
        self.local = Stage2LocalCompGCN(num_relations, hidden_dim, num_bases, dropout)
        self.cross = Stage3CrossGraphAttention(num_relations, hidden_dim, num_heads, dropout=dropout)

    @classmethod
    def from_preprocessed(cls, data: PreprocessedGoG, **kwargs: Any) -> "GoGEncoderStages1To3":
        return cls(num_entities=len(data.entities), num_types=len(data.type_id_to_index),
                   num_relations=len(data.relation_id_to_index), **kwargs)

    def forward(self, data: PreprocessedGoG) -> EncoderSliceOutput:
        device = self.initializer.entity_embedding.weight.device
        entity_types = torch.tensor(data.type_index_by_entity, dtype=torch.long, device=device)
        occurrence_entities = torch.tensor(data.entity_index_by_occurrence, dtype=torch.long, device=device)
        occurrence_graphs = torch.tensor(data.graph_index_by_occurrence, dtype=torch.long, device=device)
        b_entity, delta, z_occurrence = self.initializer(entity_types, occurrence_entities)
        root_graph_index = data.graph_id_to_index[next(g["graph_id"] for g in data.graphs if g["kind"] == "root")]

        def tensors(edges: list[Any]) -> tuple[Tensor, Tensor]:
            if not edges:
                return (torch.empty((2, 0), dtype=torch.long, device=device),
                        torch.empty((0,), dtype=torch.long, device=device))
            edge_index = torch.tensor([[e.source_occurrence_index for e in edges],
                                       [e.target_occurrence_index for e in edges]], dtype=torch.long, device=device)
            edge_type = torch.tensor([e.relation_index for e in edges], dtype=torch.long, device=device)
            return edge_index, edge_type

        local_edges = data.edge_lists["same_internal_graph"]
        local_index, local_type = tensors(local_edges)
        z_local, local_channels = self.local(z_occurrence, local_index, local_type,
                                             self.relation_embedding.weight, occurrence_graphs, root_graph_index)
        cross_edges = [*data.edge_lists["root_to_internal"], *data.edge_lists["internal_to_root"],
                       *data.edge_lists["internal_to_different_internal"]]
        cross_index, cross_type = tensors(cross_edges)
        z_cross, cross_channels, _scope_ids = self.cross(z_local, cross_index, cross_type,
                                                         self.relation_embedding.weight, occurrence_graphs,
                                                         root_graph_index)
        return EncoderSliceOutput(b_entity, delta, z_occurrence, z_local, z_cross,
                                  local_channels, cross_channels)


class GoGEncoder(nn.Module):
    """Full two-round encoder. It returns vectors; it does not score links yet."""

    def __init__(self, num_entities: int, num_types: int, num_relations: int,
                 hidden_dim: int = 256, num_bases: int = 8, num_heads: int = 4,
                 dropout: float = 0.2, num_rounds: int = 2):
        super().__init__()
        if num_rounds < 1:
            raise ValueError("num_rounds must be at least one")
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds
        self.initializer = Stage1Initializer(num_entities, num_types, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)
        self.local = Stage2LocalCompGCN(num_relations, hidden_dim, num_bases, dropout)
        self.cross = Stage3CrossGraphAttention(num_relations, hidden_dim, num_heads, dropout=dropout)
        self.readout = Stage4GraphReadout(hidden_dim, dropout)
        self.fusion = Stage5GraphEntityFusion(hidden_dim, dropout)
        self.root = Stage6RootCompGCN(num_relations, hidden_dim, num_bases, dropout)
        self.sync = Stage7IdentitySync(hidden_dim)
        self.feedback = Stage8ContextFeedback(hidden_dim, dropout=dropout)

    @classmethod
    def from_preprocessed(cls, data: PreprocessedGoG, **kwargs: Any) -> "GoGEncoder":
        return cls(num_entities=len(data.entities), num_types=len(data.type_id_to_index),
                   num_relations=len(data.relation_id_to_index), **kwargs)

    @staticmethod
    def _edge_tensors(edges: list[Any], device: torch.device) -> tuple[Tensor, Tensor]:
        if not edges:
            return (torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.long, device=device))
        return (torch.tensor([[edge.source_occurrence_index for edge in edges],
                              [edge.target_occurrence_index for edge in edges]], dtype=torch.long, device=device),
                torch.tensor([edge.relation_index for edge in edges], dtype=torch.long, device=device))

    def forward(self, data: PreprocessedGoG) -> GoGEncoderOutput:
        device = self.initializer.entity_embedding.weight.device
        entity_types = torch.tensor(data.type_index_by_entity, dtype=torch.long, device=device)
        entity_by_occ = torch.tensor(data.entity_index_by_occurrence, dtype=torch.long, device=device)
        graph_by_occ = torch.tensor(data.graph_index_by_occurrence, dtype=torch.long, device=device)
        root_graph_index = data.graph_id_to_index[next(g["graph_id"] for g in data.graphs if g["kind"] == "root")]
        internal_graph_rows = sorted((g for g in data.graphs if g["kind"] == "internal"), key=lambda g: g["graph_id"])
        internal_global_ids = [data.graph_id_to_index[g["graph_id"]] for g in internal_graph_rows]
        global_to_internal = [-1] * len(data.graphs)
        for local_idx, global_idx in enumerate(internal_global_ids):
            global_to_internal[global_idx] = local_idx
        graph_to_internal = torch.tensor(global_to_internal, dtype=torch.long, device=device)
        root_occ = torch.tensor([i for i, graph_idx in enumerate(data.graph_index_by_occurrence)
                                 if graph_idx == root_graph_index], dtype=torch.long, device=device)
        member_occ = [i for i, graph_idx in enumerate(data.graph_index_by_occurrence)
                      if graph_to_internal[graph_idx].item() >= 0]
        member_occ_tensor = torch.tensor(member_occ, dtype=torch.long, device=device)
        member_graph = graph_to_internal[graph_by_occ[member_occ_tensor]]
        owner_root = torch.tensor(data.owner_root_by_graph, dtype=torch.long, device=device)
        root_occ_by_entity = torch.tensor(data.root_occurrence_by_entity, dtype=torch.long, device=device)
        owned_graph_by_entity = torch.tensor(data.owned_graph_by_entity, dtype=torch.long, device=device)
        has_owned_graph = torch.tensor(data.has_owned_graph, dtype=torch.bool, device=device)
        has_owner_root = torch.tensor(data.has_owner_root, dtype=torch.bool, device=device)
        graph_nonempty = torch.zeros(len(internal_graph_rows), dtype=torch.bool, device=device)
        if member_graph.numel():
            graph_nonempty[member_graph.unique()] = True

        b_entity, _, z = self.initializer(entity_types, entity_by_occ)
        h_graph = z.new_zeros((len(internal_graph_rows), self.hidden_dim))
        rounds: list[EncoderRoundOutput] = []
        cross_edges = [*data.edge_lists["root_to_internal"], *data.edge_lists["internal_to_root"],
                       *data.edge_lists["internal_to_different_internal"]]
        root_edges = data.edge_lists["root_to_root"]
        local_edges = data.edge_lists["same_internal_graph"]
        local_index, local_type = self._edge_tensors(local_edges, device)
        cross_index, cross_type = self._edge_tensors(cross_edges, device)
        root_index, root_type = self._edge_tensors(root_edges, device)

        for _ in range(self.num_rounds):
            z_local, local_count = self.local(z, local_index, local_type, self.relation_embedding.weight,
                                              graph_by_occ, root_graph_index)
            z_cross, cross_count, _ = self.cross(z_local, cross_index, cross_type,
                                                 self.relation_embedding.weight, graph_by_occ, root_graph_index)
            h_new, _, _ = self.readout(z_cross, member_graph, member_occ_tensor, h_graph)
            z_owner, h_fused, _ = self.fusion(z_cross, h_new, owner_root, graph_nonempty)
            z_root, root_count = self.root(z_owner, root_occ, root_index, root_type,
                                           self.relation_embedding.weight)
            b_new, _ = self.sync(b_entity, z_root, entity_by_occ)
            z_next, _ = self.feedback(b_new, z_root, h_fused, entity_by_occ, graph_by_occ,
                                      graph_to_internal, root_occ_by_entity, owned_graph_by_entity,
                                      has_owned_graph, owner_root, has_owner_root, graph_nonempty,
                                      root_graph_index)
            rounds.append(EncoderRoundOutput(z_local, z_cross, h_new, z_owner, z_root, b_new,
                                             z_next, local_count, cross_count, root_count))
            b_entity, z, h_graph = b_new, z_next, h_fused
        return GoGEncoderOutput(b_entity, z, h_graph, rounds)


__all__ = ["Stage1Initializer", "Stage2LocalCompGCN", "Stage3CrossGraphAttention",
           "Stage4GraphReadout", "Stage5GraphEntityFusion", "Stage6RootCompGCN",
           "Stage7IdentitySync", "Stage8ContextFeedback", "GoGEncoderStages1To3",
           "ComplExDecoder", "EncoderSliceOutput", "GoGEncoder",
           "GoGEncoderOutput", "EncoderRoundOutput"]
