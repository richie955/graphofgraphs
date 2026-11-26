"""Memory-bounded helpers shared by relation-based message-passing stages."""

import torch
from torch import Tensor


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
