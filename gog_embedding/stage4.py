"""Stage 4: make one summary vector for each internal graph."""

import torch
from torch import Tensor, nn


class Stage4GraphReadout(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.graph_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.attention_vector = nn.Parameter(torch.empty(hidden_dim))
        self.candidate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.attention_vector, std=hidden_dim ** -0.5)

    def forward(self, z_cross: Tensor, member_graph_index: Tensor, member_occurrence_index: Tensor,
                h_graph_prev: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n_graphs = h_graph_prev.shape[0]
        if member_occurrence_index.numel() == 0:
            return h_graph_prev, z_cross.new_empty((0,)), z_cross.new_empty((0,))
        graph_idx = member_graph_index.to(z_cross.device, dtype=torch.long)
        occ_idx = member_occurrence_index.to(z_cross.device, dtype=torch.long)
        if torch.any(graph_idx < 0) or torch.any(graph_idx >= n_graphs):
            raise ValueError("Stage 4 graph membership index is out of range")
        if torch.any(occ_idx < 0) or torch.any(occ_idx >= z_cross.shape[0]):
            raise ValueError("Stage 4 occurrence membership index is out of range")
        z = z_cross[occ_idx]
        h = h_graph_prev[graph_idx]
        hidden = torch.tanh(self.node_score(z) + self.graph_score(h) + self.attention_bias)
        scores = hidden @ self.attention_vector
        maxima = scores.new_full((n_graphs,), -torch.inf)
        maxima.scatter_reduce_(0, graph_idx, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - maxima[graph_idx])
        denom = scores.new_zeros((n_graphs,))
        denom.index_add_(0, graph_idx, weights)
        attention = weights / denom[graph_idx].clamp_min(torch.finfo(scores.dtype).tiny)
        pooled = z_cross.new_zeros((n_graphs, self.hidden_dim))
        pooled.index_add_(0, graph_idx, attention[:, None] * z)
        active = torch.zeros(n_graphs, dtype=torch.bool, device=z_cross.device)
        active[graph_idx] = True
        joined = torch.cat((h_graph_prev, pooled), dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        gate = torch.sigmoid(self.gate(joined))
        updated = self.norm(h_graph_prev + self.dropout(gate * candidate))
        h_new = torch.where(active[:, None], updated, h_graph_prev)
        return h_new, attention, pooled
