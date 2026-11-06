# Stage 7 — Global Identity Synchronization

**Purpose:** Collect context-specific information from an entity's observed GraphNode occurrences and update its single global canonical state once per propagation round.

**Inputs:** Stage 6 occurrence states and current canonical entity states  
**Output:** One updated canonical state per global entity  
**Default:** Shared occurrence attention + projected message aggregation + GRUCell

---

## 1. Responsibility and boundary

The same real-world entity can appear in ROOT and in several internal graph contexts. These occurrences must influence one shared canonical identity without collapsing their contextual states into duplicate trainable embeddings.

Stage 7 performs that synchronization after local/cross-graph propagation, graph/entity fusion, and ROOT propagation. For each entity (v), it:

1. Scores the active occurrence contexts for (v).
2. Aggregates a weighted occurrence message.
3. Uses a GRU update to revise the single canonical vector (b_v).

It does not overwrite all occurrence states with the new canonical state. Stage 8 recomputes contextual adjustments from the updated canonical states and feeds context back down.

Do not separately add the graph vector (h_G) as an extra occurrence in this stage: Stage 5 has already fused graph information into the owner's ROOT occurrence, and the graph state will be used by Stage 8. This avoids counting the same graph summary twice.

## 2. Inputs and outputs

| Input | Shape | Meaning |
|---|---:|---|
| `b_entity` | `[N_entity, 256]` | Canonical states from initialization or prior round |
| `z_after_root` | `[N_occurrence, 256]` | Occurrence states after Stage 6 |
| `entity_index_by_occurrence` | `[N_occurrence]` | Maps each occurrence to one canonical entity row |
| `occurrence_active_mask` | `[N_occurrence]` | Selects occurrences available to this encoder pass |
| `b_entity_new` | `[N_entity, 256]` | Updated canonical state; absent entities retain prior state |
| `sync_attention` | ragged by entity | Attention weights over that entity's active occurrences |

Internal occurrence rows in `z_after_root` are unchanged by Stage 6; ROOT rows have the Stage 6 update. Both can contribute to synchronization.

## 3. Attention over occurrence contexts

For each active occurrence (o=(v,G)), let (z_{v,G}\in\mathbb R^{256}) be its current state. The canonical state (b_v\in\mathbb R^{256}) acts as the query/context for deciding how much this occurrence should influence the global identity.

Compute an occurrence score using one shared additive scorer:

\[
\ell_{v,G}=w_s^\top\tanh(W_z z_{v,G}+W_b b_v+b_s),
\]

where (W_z,W_b\in\mathbb R^{256\times256}) and (w_s,b_s\in\mathbb R^{256}).

Normalize only over active occurrences that share the same canonical entity ID:

\[
\boxed{\alpha_{v,G}=\operatorname{softmax}_{G\in\mathcal O(v)}(\ell_{v,G})}.
\]

Here \(\mathcal O(v)\) is the set of observed active GraphNode occurrences for entity (v). This is a segmented softmax grouped by canonical entity index; there is no attention across different entities.

## 4. Occurrence message and aggregation

Project each occurrence state into a synchronization message:

\[
m_{v,G}=W_m z_{v,G}+b_m,
\qquad W_m\in\mathbb R^{256\times256},\quad b_m\in\mathbb R^{256}.
\]

Aggregate the messages using the learned occurrence attention weights:

\[
\boxed{u_v=\sum_{G\in\mathcal O(v)}\alpha_{v,G}m_{v,G}}.
\]

The aggregation is permutation-invariant over occurrences. It uses only occurrences present in the graph data for this pass; it does not create a row for every possible entity/graph pair.

## 5. GRU canonical update

Use a shared GRU cell with the aggregated occurrence message as input and the previous canonical state as hidden state:

\[
\boxed{b_v^{new}=\operatorname{GRUCell}(u_v,b_v)}.
\]

Both input and hidden dimensions are 256. The GRU parameters are shared across every entity. This lets the canonical state retain useful information from the previous round while incorporating context chosen from its occurrences.

Entities with one active occurrence use that occurrence's message with attention weight 1. Entities with no active occurrence in the current encoder pass keep their prior canonical state:

\[
b_v^{new}=b_v\quad\text{if }\mathcal O(v)=\varnothing.
\]

Do not apply an arbitrary zero message to absent entities through the GRU; a GRU with zero input could still change its hidden state due to gates and biases.

## 6. Synchronous, grouped implementation

All occurrence messages and attention scores in a round must be calculated from the same snapshots `z_after_root` and `b_entity`. Then update each canonical entity once. Do not update entities sequentially while iterating over occurrences.

```python
active_occ = occurrence_active_mask.nonzero().flatten()
entity_ids = entity_index_by_occurrence[active_occ]
z = z_after_root[active_occ]
b = b_entity[entity_ids]

score_hidden = tanh(
    occurrence_score_linear(z)
    + canonical_query_linear(b)
    + score_bias
)
scores = score_hidden @ score_vector
alpha = segmented_softmax(scores, entity_ids)

messages = occurrence_message_linear(z)
aggregated = segment_sum(
    alpha[:, None] * messages,
    entity_ids,
    num_segments=N_entity,
)
active_entity_ids = unique(entity_ids)
updated = gru_cell(
    aggregated[active_entity_ids],
    b_entity[active_entity_ids],
)

b_entity_new = b_entity.index_copy(0, active_entity_ids, updated)
```

Use stable segmented operations. The model may remap active entity IDs to a compact segment range, provided it maps updated states back to the correct canonical rows. If minibatches are used, occurrences belonging to the same canonical entity must be synchronized together within the pass, or their messages must be accumulated before the single GRU update. Do not run multiple order-dependent GRU updates for one entity in one round.

## 7. State and normalization convention

Stage 1 defines occurrence states as:

\[
z_{v,G}=\operatorname{LayerNorm}(b_v+\Delta_{v,G}).
\]

Stage 7's GRU output becomes the new canonical state (b_v^{new}). Keep it as the canonical hidden state; Stage 8 applies the configured occurrence construction/normalization path when recomputing (z_{v,G}). Do not introduce per-occurrence persistent parameters during synchronization.

If a future design adds LayerNorm to the canonical GRU output, apply it consistently to all entities and document that as an architecture change; it is not an extra per-entity normalization choice.

## 8. Occurrence membership and duplicate handling

- Group by `Entity.entity_id` through `entity_index_by_occurrence`, not by name strings.
- One `GraphNode` contributes one occurrence message for its actual graph context.
- An entity appearing in ROOT and two internal graphs contributes three context rows, all to one canonical segment.
- Preserve multiple GraphNode rows in one graph only if the data model says they are distinct valid occurrences; otherwise resolve duplicates before encoding.
- Owner graph vectors are not separate occurrence rows. Their upward contribution is already represented in the owner ROOT state through Stage 5.

## 9. Parameters and sharing

Trainable Stage 7 parameters:

- Occurrence attention projections (W_z,W_b,w_s,b_s).
- Occurrence message projection (W_m,b_m).
- GRUCell input, hidden, and bias parameters.

All are shared across entities and graph contexts. There is no entity-specific synchronizer and no graph-specific or occurrence-specific trainable lookup vector. The global canonical identity table remains the only per-entity embedding table.

## 10. Gradient flow and leakage controls

Link-prediction gradients flow through the GRU update, attention weights, occurrence message projection, and all previous occurrence encoders. Do not detach (u_v), (m_{v,G}), or the occurrence states during training.

The occurrence states must be produced from the allowed training message-passing graph. A held-out relation may still be a prediction target, but it and its generated reverse channel must not have affected its endpoint occurrence states. Stage 7 must not use held-out relation labels or validation/test neighborhoods to assign attention weights.

## 11. Edge cases

| Case | Behavior |
|---|---|
| Entity with one active occurrence | Attention weight 1; GRU updates from that message |
| Entity with several occurrences | Softmax and weighted message aggregation over only that entity's occurrences |
| Entity with no active occurrence in sampled pass | Keep prior canonical state exactly |
| Entity appears in ROOT and internal graphs | Aggregate all its active occurrence contexts into one canonical state |
| Graph-valued entity owner | Root occurrence carries Stage 5 graph information; do not separately pool the same graph vector here |
| Multiple occurrences of same entity in one graph | Keep or deduplicate according to the explicit data contract; never silently merge different occurrence IDs |
| All entities absent from active occurrence batch | Return `b_entity` unchanged |
| Minibatch splits one entity's occurrences | Combine messages first, apply one GRU update per entity |

## 12. Acceptance criteria

Stage 7 is correct when:

1. There remains exactly one canonical state per global entity.
2. All active GraphNode occurrences for an entity are grouped by canonical entity ID.
3. Attention normalizes over occurrences of the same entity only.
4. Occurrence messages are aggregated once per entity and passed through a shared GRUCell.
5. The previous canonical state is the GRU hidden state.
6. Entities with no active occurrences remain bitwise unchanged.
7. An entity with one occurrence is handled without special trainable parameters.
8. Graph vectors are not double-counted as extra occurrences.
9. Updates are synchronous and independent of occurrence iteration order.
10. Gradients flow to the occurrence encoders and Stage 1 canonical/features.

## 13. Handoff to Stage 8

Stage 8 receives:

```text
b_entity_new: [N_entity, 256]
h_graph_fused: [N_internal_graph, 256]
owner_feedback: [N_internal_graph, 256]
entity_index_by_occurrence: [N_occurrence]
graph_index_by_occurrence: [N_occurrence]
```

It computes graph-specific contextual adjustments from the updated canonical states and current graph/owner context, then prepares occurrence states for the next propagation round. It must preserve the single-canonical-identity contract.
