"""Stage 9: directional link scoring for source and target graph nodes."""

import torch
from torch import Tensor, nn


class ComplExDecoder(nn.Module):
    """Score directed triples with ComplEx using the two node vectors given."""

    def __init__(self, num_relations: int, hidden_dim: int = 256):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for ComplEx")
        self.hidden_dim = hidden_dim
        self.complex_dim = hidden_dim // 2
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation = nn.Embedding(num_relations, 2 * self.complex_dim)
        nn.init.xavier_uniform_(self.relation.weight)

    def _complex(self, state: Tensor) -> Tensor:
        projected = self.projection(state)
        return torch.complex(projected[..., :self.complex_dim], projected[..., self.complex_dim:])

    def project_for_ranking(self, states: Tensor) -> Tensor:
        """Project every occurrence once before ranking many candidate links."""
        return self.projection(states).reshape(*states.shape[:-1], 2, self.complex_dim)

    def score_candidate_matrix(self, projected_occurrences: Tensor, query_node_indices: Tensor,
                               relation_ids: Tensor, candidate_node_indices: Tensor,
                               side: str) -> Tensor:
        """Return [queries, candidates] scores without re-projecting states.

        ``side='tail'`` fixes each query head and ranks candidate tails;
        ``side='head'`` fixes each query tail and ranks candidate heads.
        """
        query = projected_occurrences[query_node_indices]
        candidates = projected_occurrences[candidate_node_indices]
        relation = self.relation(relation_ids).reshape(-1, 2, self.complex_dim)
        qr, qi = query[:, 0], query[:, 1]
        rr, ri = relation[:, 0], relation[:, 1]
        cr, ci = candidates[:, 0], candidates[:, 1]
        if side == "tail":
            weight_real = qr * rr - qi * ri
            weight_imag = qr * ri + qi * rr
            return weight_real @ cr.T + weight_imag @ ci.T
        if side == "head":
            weight_real = rr * qr + ri * qi
            weight_imag = rr * qi - ri * qr
            return (cr @ weight_real.T + ci @ weight_imag.T).T
        raise ValueError("side must be 'head' or 'tail'")

    def score(self, head: Tensor, relation_id: Tensor, tail: Tensor) -> Tensor:
        if head.shape != tail.shape or head.shape[-1] != self.hidden_dim:
            raise ValueError("head and tail must have matching [..., hidden_dim] shapes")
        relation_id = relation_id.to(device=head.device, dtype=torch.long)
        rel = self.relation(relation_id)
        rel_complex = torch.complex(rel[..., :self.complex_dim], rel[..., self.complex_dim:])
        h = self._complex(head)
        t = self._complex(tail)
        return (h * rel_complex * t.conj()).sum(dim=-1).real

    def score_candidates(self, head: Tensor, relation_id: Tensor, tails: Tensor) -> Tensor:
        """Score candidate tails for one fixed head and relation."""
        if tails.ndim != 2:
            raise ValueError("tails must have shape [num_candidates, hidden_dim]")
        return self.score(head.expand_as(tails), relation_id.expand(tails.shape[0]), tails)
