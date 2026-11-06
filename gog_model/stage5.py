"""Stage 5: exchange information between each graph and its owner in ROOT."""

import torch
from torch import Tensor, nn


class Stage5GraphEntityFusion(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.up_candidate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.up_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.up_norm = nn.LayerNorm(hidden_dim)
        self.down_candidate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.down_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.down_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_cross: Tensor, h_graph: Tensor, owner_root_index: Tensor,
                graph_nonempty: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        device = z_cross.device
        owner = owner_root_index.to(device=device, dtype=torch.long)
        active = graph_nonempty.to(device=device, dtype=torch.bool)
        if owner.numel() != h_graph.shape[0] or active.numel() != h_graph.shape[0]:
            raise ValueError("Stage 5 graph, owner, and non-empty mappings must align")
        active_graphs = active.nonzero(as_tuple=False).flatten()
        if active_graphs.numel() == 0:
            return z_cross, h_graph, z_cross.new_zeros((h_graph.shape[0], z_cross.shape[1]))
        owner_rows = owner[active_graphs]
        if torch.any(owner_rows < 0) or torch.any(owner_rows >= z_cross.shape[0]):
            raise ValueError("Stage 5 owner ROOT occurrence is invalid")
        if owner_rows.unique().numel() != owner_rows.numel():
            raise ValueError("Stage 5 owner ROOT occurrences must be unique")
        owner_state = z_cross[owner_rows]
        graph_state = h_graph[active_graphs]
        up_input = torch.cat((owner_state, graph_state), dim=-1)
        up_candidate = torch.tanh(self.up_candidate(up_input))
        up_gate = torch.sigmoid(self.up_gate(up_input))
        owner_up = self.up_norm(owner_state + self.dropout(up_gate * up_candidate))
        down_input = torch.cat((graph_state, owner_up), dim=-1)
        down_candidate = torch.tanh(self.down_candidate(down_input))
        down_gate = torch.sigmoid(self.down_gate(down_input))
        graph_down = self.down_norm(graph_state + self.dropout(down_gate * down_candidate))
        z_owner_all = z_cross.index_copy(0, owner_rows, owner_up)
        h_fused = h_graph.index_copy(0, active_graphs, graph_down)
        owner_feedback = h_graph.new_zeros(h_graph.shape).index_copy(0, active_graphs, owner_up)
        return z_owner_all, h_fused, owner_feedback
