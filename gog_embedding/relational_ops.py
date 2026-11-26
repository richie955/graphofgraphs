"""Memory-bounded helpers shared by relation-based message-passing stages."""

import torch
from torch import Tensor


def initialize_messages(basis: Tensor, coefficients: Tensor, mode: str) -> None:
    if mode == "legacy":
        torch.nn.init.xavier_uniform_(basis)
        torch.nn.init.xavier_uniform_(coefficients)
    elif mode == "per_matrix":
        with torch.no_grad():
            for direction in basis:
                for matrix in direction:
                    torch.nn.init.xavier_uniform_(matrix)
        torch.nn.init.normal_(coefficients, std=coefficients.shape[-1] ** -0.5)
    else:
        raise ValueError("basis_init must be legacy or per_matrix")


def relational_aggregate(z: Tensor, src: Tensor, dst: Tensor, edge_type: Tensor,
                         relations: Tensor, coefficients: Tensor, basis: Tensor,
                         composition: str, normalization: str) -> Tensor:
    """Each recorded fact generates exactly two computational channels."""
    aggregate = torch.zeros_like(z)
    if edge_type.numel() == 0:
        return aggregate
    senders, receivers = torch.cat((src, dst)), torch.cat((dst, src))
    rel_ids = edge_type.repeat(2)
    directions = torch.cat((torch.zeros_like(edge_type), torch.ones_like(edge_type)))
    vectors = relations[rel_ids]
    composed = z[senders] * (1 + vectors if composition == "residual" else vectors)
    if normalization == "relation":
        # Normalize independently for each receiver/relation/direction, without
        # allocating a dense nodes x relations tensor.
        keys = (receivers * relations.shape[0] + rel_ids) * 2 + directions
        _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
        composed = composed / counts[inverse, None].to(z.dtype)
    aggregate_basis_messages(aggregate, receivers, composed, rel_ids, directions,
                             coefficients, basis)
    if normalization == "global":
        degree = torch.bincount(receivers, minlength=z.shape[0]).to(z.dtype)
        aggregate = aggregate / degree.clamp_min(1)[:, None]
    return aggregate


def message_ratio(aggregate: Tensor, self_term: Tensor) -> float:
    with torch.no_grad():
        if aggregate.numel() == 0:
            return 0.0
        return float((aggregate.norm(dim=-1) /
                      self_term.norm(dim=-1).clamp_min(1e-12)).mean())


def aggregate_basis_messages(aggregate: Tensor, receivers: Tensor, composed: Tensor,
                             relation_ids: Tensor, directions: Tensor,
                             coefficients: Tensor, basis: Tensor) -> None:
    """Apply relation transforms in groups instead of materializing one matrix
    per edge. The old per-edge tensor scales as E x D x D; this keeps temporary
    storage near E x D plus one D x D matrix per active relation/direction.
    """
    for direction in (0, 1):
        direction_mask = directions == direction
        active_relations = torch.unique(relation_ids[direction_mask]).tolist()
        for relation_id in active_relations:
            positions = torch.nonzero(
                direction_mask & (relation_ids == relation_id), as_tuple=False
            ).flatten()
            transform = torch.einsum(
                "b,bij->ij", coefficients[relation_id, direction], basis[direction]
            )
            messages = torch.nn.functional.linear(composed[positions], transform)
            aggregate.index_add_(0, receivers[positions], messages)
