"""Stage 3: pass information across explicitly linked graph contexts."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Stage3CrossGraphAttention(nn.Module):
    """Sparse relation-conditioned multihead attention over cross-context facts."""

    SCOPE_ROOT_TO_INTERNAL = 0
    SCOPE_INTERNAL_TO_ROOT = 1
    SCOPE_INTERNAL_TO_INTERNAL = 2

    def __init__(self, num_relations: int, hidden_dim: int = 256, num_heads: int = 4,
                 scope_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scope_embedding = nn.Embedding(3, scope_dim)
        self.direction_embedding = nn.Embedding(2, scope_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_scope = nn.Linear(scope_dim, hidden_dim, bias=False)
        self.value_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_scope = nn.Linear(scope_dim, hidden_dim, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(num_relations, num_heads))
        self.scope_direction_bias = nn.Parameter(torch.zeros(3, 2, num_heads))
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _channel_scope(sender_graph: Tensor, receiver_graph: Tensor, root_graph_index: int) -> Tensor:
        sender_root = sender_graph == int(root_graph_index)
        receiver_root = receiver_graph == int(root_graph_index)
        scope = torch.full_like(sender_graph, Stage3CrossGraphAttention.SCOPE_INTERNAL_TO_INTERNAL)
        scope[sender_root & ~receiver_root] = Stage3CrossGraphAttention.SCOPE_ROOT_TO_INTERNAL
        scope[~sender_root & receiver_root] = Stage3CrossGraphAttention.SCOPE_INTERNAL_TO_ROOT
        if torch.any(sender_root & receiver_root):
            raise ValueError("Stage 3 cannot receive ROOT-to-ROOT edges")
        return scope

    @staticmethod
    def _segmented_softmax(scores: Tensor, segment: Tensor, n_segments: int) -> Tensor:
        # scores [E,H]; stable per-segment maximum then index_add denominators.
        if scores.shape[0] == 0:
            return scores
        index = segment[:, None].expand(-1, scores.shape[1])
        maxima = scores.new_full((n_segments, scores.shape[1]), -torch.inf)
        maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - maxima[segment])
        denominators = scores.new_zeros((n_segments, scores.shape[1]))
        denominators.index_add_(0, segment, weights)
        return weights / denominators[segment].clamp_min(torch.finfo(scores.dtype).tiny)

    def forward(self, z_local: Tensor, edge_index: Tensor, edge_type: Tensor, relation_vectors: Tensor,
                graph_index_by_occurrence: Tensor, root_graph_index: int) -> tuple[Tensor, int, Tensor]:
        n_nodes, dim = z_local.shape
        if dim != self.hidden_dim:
            raise ValueError("z_local hidden dimension does not match Stage 3")
        device = z_local.device
        edge_index = edge_index.to(device=device, dtype=torch.long).reshape(2, -1)
        edge_type = edge_type.to(device=device, dtype=torch.long).reshape(-1)
        graph_ids = graph_index_by_occurrence.to(device=device, dtype=torch.long)
        rels = relation_vectors.to(device=device, dtype=z_local.dtype)
        if edge_index.shape[1] != edge_type.numel():
            raise ValueError("edge_index and edge_type lengths differ")
        src, dst = edge_index
        if src.numel() and (src.min() < 0 or dst.min() < 0 or src.max() >= n_nodes or dst.max() >= n_nodes):
            raise ValueError("Stage 3 edge endpoint outside occurrence range")
        if src.numel() and torch.any(graph_ids[src] == graph_ids[dst]):
            raise ValueError("Stage 3 accepts only cross-context edges")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= rels.shape[0]):
            raise ValueError("Stage 3 relation index outside relation table")
        if src.numel() == 0:
            return z_local, 0, torch.empty((0,), dtype=torch.long, device=device)

        # Reverse channels reverse endpoints; scope is derived per actual channel.
        senders = torch.cat((src, dst))
        receivers = torch.cat((dst, src))
        rel_ids = torch.cat((edge_type, edge_type))
        directions = torch.cat((torch.zeros_like(edge_type), torch.ones_like(edge_type)))
        scope_ids = self._channel_scope(graph_ids[senders], graph_ids[receivers], root_graph_index)
        q = self.query(z_local[receivers]).view(-1, self.num_heads, self.head_dim)
        scope_features = self.scope_embedding(scope_ids) + self.direction_embedding(directions)
        key = (self.key_node(z_local[senders]) + self.key_relation(rels[rel_ids]) + self.key_scope(scope_features)).view(-1, self.num_heads, self.head_dim)
        value = (self.value_node(z_local[senders]) + self.value_relation(rels[rel_ids]) + self.value_scope(scope_features)).view(-1, self.num_heads, self.head_dim)
        scores = (q * key).sum(-1) / (self.head_dim ** 0.5)
        scores = scores + self.relation_bias[rel_ids] + self.scope_direction_bias[scope_ids, directions]
        alpha = self._segmented_softmax(scores, receivers, n_nodes)
        weighted = (alpha[..., None] * value).reshape(-1, dim)
        aggregate_heads = z_local.new_zeros((n_nodes, self.num_heads, self.head_dim))
        aggregate_heads.index_add_(0, receivers, weighted.view(-1, self.num_heads, self.head_dim))
        aggregate = self.output(aggregate_heads.reshape(n_nodes, dim))
        active = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        active[receivers] = True
        gate = torch.sigmoid(self.gate(torch.cat((z_local, aggregate), dim=-1)))
        updated = self.norm(z_local[active] + self.dropout(gate[active] * aggregate[active]))
        z_cross = z_local.index_copy(0, active.nonzero(as_tuple=False).flatten(), updated)
        return z_cross, int(senders.numel()), scope_ids
