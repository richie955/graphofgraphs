"""Stage 8: rebuild each graph node's context for the next round."""

import torch
from torch import Tensor, nn


class Stage8ContextFeedback(nn.Module):
    def __init__(self, hidden_dim: int = 256, scope_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scope_embedding = nn.Embedding(2, scope_dim)
        input_dim = 5 * hidden_dim + scope_dim + 4
        self.candidate = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, b_entity: Tensor, z_current: Tensor, h_graph_fused: Tensor,
                entity_index_by_occurrence: Tensor, graph_index_by_occurrence: Tensor,
                graph_to_internal_index: Tensor, root_occurrence_by_entity: Tensor,
                owned_graph_by_entity: Tensor, has_owned_graph: Tensor,
                owner_root_by_internal_graph: Tensor, has_graph_owner_root: Tensor,
                graph_nonempty: Tensor, root_graph_index: int) -> tuple[Tensor, Tensor]:
        device = z_current.device
        entity_idx = entity_index_by_occurrence.to(device=device, dtype=torch.long)
        graph_idx = graph_index_by_occurrence.to(device=device, dtype=torch.long)
        internal_map = graph_to_internal_index.to(device=device, dtype=torch.long)
        root_by_entity = root_occurrence_by_entity.to(device=device, dtype=torch.long)
        owned_graph = owned_graph_by_entity.to(device=device, dtype=torch.long)
        has_owned = has_owned_graph.to(device=device, dtype=torch.bool)
        owner_root = owner_root_by_internal_graph.to(device=device, dtype=torch.long)
        has_owner_root = has_graph_owner_root.to(device=device, dtype=torch.bool)
        nonempty = graph_nonempty.to(device=device, dtype=torch.bool)
        n_occ = z_current.shape[0]
        if n_occ == 0:
            return z_current, z_current

        b = b_entity[entity_idx]
        is_root = graph_idx == int(root_graph_index)
        scope_id = (~is_root).long()
        # Internal nodes use their containing graph. ROOT owners use their own graph.
        root_graph_for_entity = owned_graph[entity_idx]
        graph_for_context = torch.where(is_root, root_graph_for_entity, graph_idx)
        safe_graph = graph_for_context.clamp(min=0, max=max(internal_map.numel() - 1, 0))
        internal_idx = internal_map[safe_graph]
        valid_graph_index = (graph_for_context >= 0) & (graph_for_context < internal_map.numel())
        has_graph = valid_graph_index & (internal_idx >= 0)
        if nonempty.numel():
            safe_internal = internal_idx.clamp(min=0, max=nonempty.numel() - 1)
            has_graph = has_graph & nonempty[safe_internal]
            h_context = h_graph_fused[safe_internal] * has_graph[:, None].to(z_current.dtype)
        else:
            has_graph = torch.zeros(n_occ, dtype=torch.bool, device=device)
            h_context = z_current.new_zeros((n_occ, self.hidden_dim))

        same_indices = root_by_entity[entity_idx]
        has_same = (~is_root) & (same_indices >= 0)
        safe_same = same_indices.clamp(min=0, max=max(n_occ - 1, 0))
        same_root = z_current[safe_same] * has_same[:, None].to(z_current.dtype)

        if owner_root.numel():
            safe_internal_for_owner = internal_idx.clamp(min=0, max=owner_root.numel() - 1)
            owner_occurrences = owner_root[safe_internal_for_owner]
            has_owner = (~is_root) & (internal_idx >= 0) & has_owner_root[safe_internal_for_owner]
            deduplicated = has_owner & has_same & (owner_occurrences == same_indices)
            has_owner_context = has_owner & ~deduplicated & (owner_occurrences >= 0)
            safe_owner = owner_occurrences.clamp(min=0, max=max(n_occ - 1, 0))
            owner_feedback = z_current[safe_owner] * has_owner_context[:, None].to(z_current.dtype)
        else:
            has_owner = torch.zeros(n_occ, dtype=torch.bool, device=device)
            deduplicated = torch.zeros(n_occ, dtype=torch.bool, device=device)
            owner_feedback = z_current.new_zeros((n_occ, self.hidden_dim))

        flags = torch.stack((has_graph, has_same, has_owner, deduplicated), dim=-1).to(z_current.dtype)
        features = torch.cat((b, z_current, h_context, same_root, owner_feedback,
                              self.scope_embedding(scope_id), flags), dim=-1)
        candidate = torch.tanh(self.candidate(features))
        gate = torch.sigmoid(self.gate(features))
        delta = self.dropout(gate * candidate)
        z_next = self.norm(b + delta)
        return z_next, delta
