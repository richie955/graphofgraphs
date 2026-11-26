"""Simple public entry point for the GoG model stages."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from data_pipeline.preprocessing import PreprocessedGoG
from .stage1 import Stage1Initializer
from .stage2 import Stage2LocalCompGCN
from .stage3 import Stage3CrossGraphAttention
from .stage4 import Stage4GraphReadout
from .stage5 import Stage5GraphEntityFusion
from .stage6 import Stage6RootCompGCN
from .stage7 import Stage7IdentitySync
from .stage8 import Stage8ContextFeedback
from .stage9 import ComplExDecoder
from .structural import StructuralCategories, FamilyGraphReadout


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
                 dropout: float = 0.2, num_rounds: int | None = None,
                 profile: str = "legacy", feature_mode: str = "auto",
                 basis_init: str | None = None, composition: str | None = None,
                 normalization: str | None = None, identity_residual: bool = True,
                 local_enabled: bool = True, cross_enabled: bool = True,
                 fusion_enabled: bool = True, root_enabled: bool = True,
                 sync_enabled: bool = False, feedback_enabled: bool = False,
                 family_readout: bool = True):
        super().__init__()
        if profile not in {"legacy", "structural"}:
            raise ValueError("profile must be legacy or structural")
        num_rounds = num_rounds if num_rounds is not None else (2 if profile == "legacy" else 1)
        if num_rounds < 1:
            raise ValueError("num_rounds must be at least one")
        if feature_mode not in {"auto", "id_type", "categorical"}:
            raise ValueError("invalid feature_mode")
        if profile == "structural" and ((sync_enabled and not feedback_enabled)
                                         or ((sync_enabled or feedback_enabled) and num_rounds < 2)):
            raise ValueError("synchronization requires feedback; refinement requires at least two rounds")
        self.profile = profile
        self.local_enabled, self.cross_enabled = local_enabled, cross_enabled
        self.fusion_enabled, self.root_enabled = fusion_enabled, root_enabled
        self.sync_enabled, self.feedback_enabled = sync_enabled, feedback_enabled
        self.family_readout = family_readout
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds
        self.initializer = Stage1Initializer(num_entities, num_types, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)
        message_options = dict(basis_init=basis_init or ("legacy" if profile == "legacy" else "per_matrix"),
                               composition=composition or ("multiply" if profile == "legacy" else "residual"),
                               normalization=normalization or ("global" if profile == "legacy" else "relation"))
        self.local = Stage2LocalCompGCN(num_relations, hidden_dim, num_bases, dropout, **message_options)
        self.cross = Stage3CrossGraphAttention(num_relations, hidden_dim, num_heads, dropout=dropout)
        self.readout = Stage4GraphReadout(hidden_dim, dropout)
        self.fusion = Stage5GraphEntityFusion(hidden_dim, dropout)
        self.root = Stage6RootCompGCN(num_relations, hidden_dim, num_bases, dropout, **message_options)
        self.sync = Stage7IdentitySync(hidden_dim)
        self.feedback = Stage8ContextFeedback(hidden_dim, dropout=dropout)
        if profile == "structural":
            self.chemistry = StructuralCategories(hidden_dim) if feature_mode != "id_type" else None
            self.structural_readout = FamilyGraphReadout(hidden_dim, dropout) if family_readout else None
            self.identity_scale = nn.Parameter(torch.zeros(num_types)) if identity_residual else None
            self.feedback_scale = nn.Parameter(torch.tensor(0.1))

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
                      if global_to_internal[graph_idx] >= 0]
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
        z_initial = z
        if self.profile == "structural" and self.chemistry is not None:
            b_entity = b_entity + self.chemistry(data)
            z = self.initializer.occurrence_norm(b_entity[entity_by_occ])
        entity_types_by_id = {e["entity_id"]: str(e.get("type", e.get("type_id", ""))).lower()
                              for e in data.entities}
        families = torch.tensor([{"drug": 0, "pathway": 1}.get(
            entity_types_by_id[g["owner_entity_id"]], 2) for g in internal_graph_rows],
            device=device, dtype=torch.long)
        h_graph = z.new_zeros((len(internal_graph_rows), self.hidden_dim))
        rounds: list[EncoderRoundOutput] = []
        self.last_round_changes = []
        cross_edges = [*data.edge_lists["root_to_internal"], *data.edge_lists["internal_to_root"],
                       *data.edge_lists["internal_to_different_internal"]]
        root_edges = data.edge_lists["root_to_root"]
        local_edges = data.edge_lists["same_internal_graph"]
        local_index, local_type = self._edge_tensors(local_edges, device)
        cross_index, cross_type = self._edge_tensors(cross_edges, device)
        root_index, root_type = self._edge_tensors(root_edges, device)

        for round_index in range(self.num_rounds):
            z_local, local_count = (self.local(z, local_index, local_type, self.relation_embedding.weight,
                                              graph_by_occ, root_graph_index)
                                    if self.local_enabled else (z, 0))
            if self.cross_enabled:
                z_cross, cross_count, _ = self.cross(z_local, cross_index, cross_type,
                                                     self.relation_embedding.weight, graph_by_occ, root_graph_index)
            else:
                z_cross, cross_count = z_local, 0
            if self.profile == "structural" and self.structural_readout is not None:
                h_new = self.structural_readout(z_cross, member_graph, member_occ_tensor, h_graph, families)
            else:
                h_new, _, _ = self.readout(z_cross, member_graph, member_occ_tensor, h_graph)
            if self.fusion_enabled:
                z_owner, h_fused, _ = self.fusion(z_cross, h_new, owner_root, graph_nonempty,
                                                 downward=self.profile == "legacy" or self.feedback_enabled)
            else:
                z_owner, h_fused = z_cross, h_new
            z_root, root_count = (self.root(z_owner, root_occ, root_index, root_type,
                                            self.relation_embedding.weight)
                                  if self.root_enabled else (z_owner, 0))
            b_new, z_next = b_entity, z_root
            if self.profile == "legacy" or (self.feedback_enabled and round_index + 1 < self.num_rounds):
                if self.profile == "legacy" or self.sync_enabled:
                    mask = None if self.profile == "legacy" else (torch.bincount(
                        entity_by_occ, minlength=b_entity.shape[0])[entity_by_occ] > 1)
                    b_new, _ = self.sync(b_entity, z_root, entity_by_occ, active_mask=mask)
                candidate, _ = self.feedback(b_new, z_root, h_fused, entity_by_occ, graph_by_occ,
                                             graph_to_internal, root_occ_by_entity, owned_graph_by_entity,
                                             has_owned_graph, owner_root, has_owner_root, graph_nonempty,
                                             root_graph_index)
                z_next = (candidate if self.profile == "legacy" else
                          z_root + self.feedback_scale * (candidate - z_root))
            with torch.no_grad():
                def change(before: Tensor, after: Tensor) -> float:
                    return float((after - before).norm(dim=-1).mean()) if after.shape[0] else 0.0
                self.last_round_changes.append({"local": change(z, z_local),
                                                "cross": change(z_local, z_cross),
                                                "readout": change(h_graph, h_new),
                                                "owner": change(z_cross, z_owner),
                                                "root": change(z_owner[root_occ], z_root[root_occ]),
                                                "synchronization": change(b_entity, b_new),
                                                "feedback": change(z_root, z_next)})
            rounds.append(EncoderRoundOutput(z_local, z_cross, h_new, z_owner, z_root, b_new,
                                             z_next, local_count, cross_count, root_count))
            b_entity, z, h_graph = b_new, z_next, h_fused
        if self.profile == "structural" and self.identity_scale is not None:
            scale = self.identity_scale[entity_types[entity_by_occ[root_occ]]][:, None]
            decoded_root = z_initial[root_occ] + scale * (z[root_occ] - z_initial[root_occ])
            z = z.index_copy(0, root_occ, decoded_root)
        return GoGEncoderOutput(b_entity, z, h_graph, rounds)


__all__ = ["Stage1Initializer", "Stage2LocalCompGCN", "Stage3CrossGraphAttention",
           "Stage4GraphReadout", "Stage5GraphEntityFusion", "Stage6RootCompGCN",
           "Stage7IdentitySync", "Stage8ContextFeedback", "GoGEncoderStages1To3",
           "ComplExDecoder", "EncoderSliceOutput", "GoGEncoder",
           "GoGEncoderOutput", "EncoderRoundOutput"]
