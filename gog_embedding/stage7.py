"""Stage 7: combine an entity's graph appearances into one shared vector."""

import torch
from torch import Tensor, nn


class Stage7IdentitySync(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.occurrence_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.entity_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.score_vector = nn.Parameter(torch.empty(hidden_dim))
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        nn.init.normal_(self.score_vector, std=hidden_dim ** -0.5)

    def forward(self, b_entity: Tensor, z_after_root: Tensor, entity_index_by_occurrence: Tensor,
                active_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        device = b_entity.device
        entity_idx = entity_index_by_occurrence.to(device=device, dtype=torch.long)
        active = torch.ones(entity_idx.shape[0], dtype=torch.bool, device=device) if active_mask is None else active_mask.to(device=device, dtype=torch.bool)
        if entity_idx.numel() != z_after_root.shape[0] or active.numel() != z_after_root.shape[0]:
            raise ValueError("Stage 7 occurrence mappings and states must align")
        rows = active.nonzero(as_tuple=False).flatten()
        attention_full = z_after_root.new_zeros((z_after_root.shape[0],))
        if rows.numel() == 0:
            return b_entity, attention_full
        ids = entity_idx[rows]
        z = z_after_root[rows]
        if torch.any(ids < 0) or torch.any(ids >= b_entity.shape[0]):
            raise ValueError("Stage 7 entity index is out of range")
        hidden = torch.tanh(self.occurrence_score(z) + self.entity_query(b_entity[ids]) + self.score_bias)
        score = hidden @ self.score_vector
        maxima = score.new_full((b_entity.shape[0],), -torch.inf)
        maxima.scatter_reduce_(0, ids, score, reduce="amax", include_self=True)
        exp = torch.exp(score - maxima[ids])
        denom = score.new_zeros((b_entity.shape[0],))
        denom.index_add_(0, ids, exp)
        alpha = exp / denom[ids].clamp_min(torch.finfo(score.dtype).tiny)
        attention_full = attention_full.index_copy(0, rows, alpha)
        messages = self.message(z)
        aggregate = b_entity.new_zeros(b_entity.shape)
        aggregate.index_add_(0, ids, alpha[:, None] * messages)
        active_entities = ids.unique(sorted=True)
        updated = self.update(aggregate[active_entities], b_entity[active_entities])
        return b_entity.index_copy(0, active_entities, updated), attention_full
