"""Stage 6: pass relation messages between ROOT nodes only."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relational_ops import initialize_messages, relational_aggregate, message_ratio


class Stage6RootCompGCN(nn.Module):
    def __init__(self, num_relations: int, hidden_dim: int = 256, num_bases: int = 8, dropout: float = 0.2,
                 basis_init: str = "legacy", composition: str = "multiply", normalization: str = "global"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.basis = nn.Parameter(torch.empty(2, num_bases, hidden_dim, hidden_dim))
        self.coefficients = nn.Parameter(torch.empty(num_relations, 2, num_bases))
        self.self_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        if composition not in {"multiply", "residual"} or normalization not in {"global", "relation"}:
            raise ValueError("invalid message composition or normalization")
        self.composition, self.normalization = composition, normalization
        self.last_message_self_ratio = 0.0
        initialize_messages(self.basis, self.coefficients, basis_init)

    def forward(self, z_owner_all: Tensor, root_occurrence_indices: Tensor, root_edge_index_global: Tensor,
                edge_type: Tensor, relation_vectors: Tensor) -> tuple[Tensor, int]:
        device = z_owner_all.device
        root_idx = root_occurrence_indices.to(device=device, dtype=torch.long)
        edge_index = root_edge_index_global.to(device=device, dtype=torch.long).reshape(2, -1)
        edge_type = edge_type.to(device=device, dtype=torch.long).reshape(-1)
        rels = relation_vectors.to(device=device, dtype=z_owner_all.dtype)
        if edge_index.shape[1] != edge_type.numel():
            raise ValueError("Stage 6 edge and relation counts differ")
        if root_idx.numel() == 0:
            if edge_type.numel():
                raise ValueError("Stage 6 received ROOT facts without ROOT nodes")
            return z_owner_all, 0
        to_local = torch.full((z_owner_all.shape[0],), -1, dtype=torch.long, device=device)
        to_local[root_idx] = torch.arange(root_idx.numel(), device=device)
        src_global, dst_global = edge_index
        if edge_type.numel():
            src, dst = to_local[src_global], to_local[dst_global]
            if torch.any(src < 0) or torch.any(dst < 0):
                raise ValueError("Stage 6 accepts ROOT-to-ROOT facts only")
        else:
            src = dst = edge_type
        z_root = z_owner_all[root_idx]
        aggregate = relational_aggregate(z_root, src, dst, edge_type, rels, self.coefficients,
                                         self.basis, self.composition, self.normalization)
        self_term = self.self_linear(z_root)
        self.last_message_self_ratio = message_ratio(aggregate, self_term)
        update = F.relu(self_term + aggregate)
        z_root_new = self.norm(z_root + self.dropout(update))
        return z_owner_all.index_copy(0, root_idx, z_root_new), int(edge_type.numel() * 2)
