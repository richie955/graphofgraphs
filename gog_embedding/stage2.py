"""Stage 2: pass information between nodes in the same internal graph."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relational_ops import initialize_messages, relational_aggregate, message_ratio


class Stage2LocalCompGCN(nn.Module):
    """Sparse relation-aware message passing within internal graphs only."""

    def __init__(self, num_relations: int, hidden_dim: int = 256, num_bases: int = 8, dropout: float = 0.2,
                 basis_init: str = "legacy", composition: str = "multiply", normalization: str = "global"):
        super().__init__()
        if hidden_dim < 1 or num_bases < 1 or num_relations < 1:
            raise ValueError("hidden_dim, num_bases, and num_relations must be positive")
        self.hidden_dim = hidden_dim
        self.num_bases = num_bases
        self.dropout = nn.Dropout(dropout)
        self.basis = nn.Parameter(torch.empty(2, num_bases, hidden_dim, hidden_dim))
        self.coefficients = nn.Parameter(torch.empty(num_relations, 2, num_bases))
        self.self_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        if composition not in {"multiply", "residual"} or normalization not in {"global", "relation"}:
            raise ValueError("invalid message composition or normalization")
        self.composition, self.normalization = composition, normalization
        self.last_message_self_ratio = 0.0
        initialize_messages(self.basis, self.coefficients, basis_init)

    def forward(self, z_in: Tensor, edge_index: Tensor, edge_type: Tensor, relation_vectors: Tensor,
                graph_index_by_occurrence: Tensor, root_graph_index: int) -> tuple[Tensor, int]:
        n_nodes, dim = z_in.shape
        if dim != self.hidden_dim:
            raise ValueError("z_in hidden dimension does not match Stage 2")
        device = z_in.device
        edge_index = edge_index.to(device=device, dtype=torch.long).reshape(2, -1)
        edge_type = edge_type.to(device=device, dtype=torch.long).reshape(-1)
        graph_ids = graph_index_by_occurrence.to(device=device, dtype=torch.long)
        rels = relation_vectors.to(device=device, dtype=z_in.dtype)
        if edge_index.shape[1] != edge_type.numel():
            raise ValueError("edge_index and edge_type lengths differ")
        internal_mask = graph_ids != int(root_graph_index)
        src, dst = edge_index
        if src.numel() and (src.min() < 0 or dst.min() < 0 or src.max() >= n_nodes or dst.max() >= n_nodes):
            raise ValueError("Stage 2 edge endpoint outside occurrence range")
        if src.numel() and (torch.any(~internal_mask[src]) or torch.any(~internal_mask[dst]) or torch.any(graph_ids[src] != graph_ids[dst])):
            raise ValueError("Stage 2 accepts only same-internal-graph edges")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= rels.shape[0]):
            raise ValueError("Stage 2 relation index outside relation table")

        aggregate = relational_aggregate(z_in, src, dst, edge_type, rels, self.coefficients,
                                         self.basis, self.composition, self.normalization)
        self_term = self.self_linear(z_in)
        self.last_message_self_ratio = message_ratio(aggregate[internal_mask], self_term[internal_mask])
        update = F.relu(self_term + aggregate)
        candidate = self.norm(z_in + self.dropout(update))
        z_local = torch.where(internal_mask[:, None], candidate, z_in)
        return z_local, int(src.numel() * 2)
