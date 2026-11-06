# Stage 8 — Contextual Adjustment and Downward Feedback

**Purpose:** Rebuild context-specific occurrence states from the newly synchronized canonical entity states, while carrying ROOT and internal-graph context into the next propagation round.

**Inputs:** Stage 7 canonical states, final occurrence states of the current round, fused internal graph states, ROOT and owner mappings  
**Outputs:** Next-round occurrence adjustments Δ and occurrence states (z)  
**Default:** Shared gated context network, 256D states, one shared module for all occurrences

---

## 1. Responsibility and boundary

Stage 8 closes a propagation round. Stage 7 has updated one canonical state (b_v) per entity from its active occurrences. Stage 8 now computes a graph-specific adjustment for every observed occurrence and reconstructs:

\[
\boxed{z_{v,G}^{next}=\operatorname{LayerNorm}(b_v^{new}+\Delta_{v,G}^{next})}.
\]

The adjustment is computed by shared model parameters from current context; it is not a trainable lookup row for `(entity_id, graph_id)` or `node_id`.

This stage carries information downward in two ways:

1. The updated canonical state is available to all occurrences of the same global entity.
2. The current ROOT context of an entity and the owner ROOT context of an internal graph can feed into the graph's internal occurrence states for the next round.

Stage 8 does not create relations or modify the data graph. Identity-aligned feedback is an encoder operation, not a new KG fact.

## 2. Required state for each occurrence

For each active occurrence (o=(v,G)), construct the following feature inputs:

| Feature | Meaning | If unavailable |
|---|---|---|
| (b_v^{new}) | Updated canonical state from Stage 7 | Not applicable |
| (z_o^{current}) | Current occurrence state after Stage 6; ROOT rows include ROOT propagation | Not applicable |
| (h_G^{context}) | Fused state of the internal graph containing (o); for a ROOT owner occurrence, use the graph it owns if it has one | Zero vector plus `has_graph_context=0` |
| (r_v^{ROOT}) | Current ROOT occurrence state of the same canonical entity, if (o) is internal and the entity has a ROOT occurrence | Zero vector plus `has_same_entity_root=0` |
| (r_{owner(G)}^{ROOT}) | Current ROOT state of the owner of (G), for internal occurrences in an owned graph | Zero vector plus `has_owner_root=0` |
| `scope` | Shared categorical embedding for ROOT vs internal occurrence | Always present |

For a ROOT occurrence, do not feed the same ROOT state again as (r_v^{ROOT}); its current state is already (z_o^{current}). For an internal occurrence, the same-entity ROOT context and the containing graph's owner ROOT context may both be present. If both mappings resolve to the same occurrence, include that state once and set a deduplication flag rather than double-counting it.

These feedback vectors are gathered from the same completed-round snapshots. They are not created by synthesizing or copying a KG edge.

## 3. Shared gated context network

Let (d=256). Concatenate the five vector inputs and a shared scope embedding:

\[
x_o=[b_v^{new}\Vert z_o^{current}\Vert h_G^{context}
\Vert r_v^{ROOT}\Vert r_{owner(G)}^{ROOT}
\Vert e_{scope}\Vert m_o],
\]

where (e_{scope}\in\mathbb R^{d_s}) is a small learned ROOT/internal scope embedding, and (m_o) contains binary availability flags such as `has_graph_context`, `has_same_entity_root`, `has_owner_root`, and `feedback_deduplicated`. The vector length is (5d+d_s+4) for the flags listed here.

Use shared candidate and gate projections:

\[
c_o=\tanh(W_c x_o+b_c),
\]

\[
g_o=\sigma(W_g x_o+b_g),
\]

where (W_c,W_g\in\mathbb R^{d\times(5d+d_s+4)}) and (b_c,b_g\in\mathbb R^d).

Create a gated contextual adjustment:

\[
\boxed{\Delta_o^{next}=\operatorname{Dropout}(g_o\odot c_o)}
\]

and reconstruct the occurrence state:

\[
\boxed{z_o^{next}=\operatorname{LayerNorm}(b_v^{new}+\Delta_o^{next})}.
\]

The gate can retain, suppress, or transform the context features for each embedding dimension. The canonical state provides the common identity component; the learned adjustment carries graph-specific/context-specific information.

Dropout is 0.2 by default. Use one shared LayerNorm and shared candidate/gate networks for all occurrences; no parameters are indexed by graph or occurrence ID.

## 4. Which feedback reaches which occurrence?

### ROOT occurrence

- (z_o^{current}) carries Stage 3 cross-context messages, Stage 5 owner fusion when this entity owns a non-empty internal graph, and Stage 6 ROOT relation messages.
- If this ROOT entity owns an internal graph, `h_graph_context` is that graph's fused state.
- Same-entity ROOT feedback is absent because it would duplicate (z_o^{current}).

### Internal occurrence

- (z_o^{current}) carries Stage 2 local information and Stage 3 explicit cross-context information.
- `h_graph_context` is the fused representation of the internal graph containing this occurrence.
- `same_entity_root` carries the ROOT state for this same canonical entity when such an occurrence exists. This is identity-aligned feedback.
- `owner_root` carries the updated ROOT state of the internal graph's owner, making ROOT-level information available within the owned graph.

The owner entity is not inserted into the internal graph as a member. Its ROOT state is a context input to the shared feedback module only.

## 5. Occurrence and graph mapping requirements

Preprocessing/model state must provide:

```text
entity_index_by_occurrence: [N_occurrence]
graph_index_by_occurrence:  [N_occurrence]
root_occurrence_by_entity:  entity index -> unique ROOT occurrence index, if present
owner_entity_by_graph:      internal graph index -> canonical owner entity index
owner_root_by_graph:        internal graph index -> owner ROOT occurrence index
owned_graph_by_entity:      canonical owner entity index -> internal graph index, if present
```

Use explicit masks for missing ROOT occurrences, owner mappings, and graph states. Do not use a real occurrence index such as `0` as the missing sentinel. For each occurrence, gather the exact same-entity and graph-owner context from these mappings; keep both feature values and presence flags aligned.

## 6. Vectorized implementation outline

```python
# Current-round snapshots; do not mutate these while gathering feedback.
b = b_entity_new[entity_index_by_occurrence]
z_prev = z_after_root
scope = scope_embedding(is_root_occurrence)

h_context, has_graph_context = gather_occurrence_graph_context(
    graph_index_by_occurrence,
    owned_graph_by_entity,
    h_graph_fused,
)
r_same, has_same_root = gather_same_entity_root_context(
    entity_index_by_occurrence,
    root_occurrence_by_entity,
    z_after_root,
    is_root_occurrence,
)
r_owner, has_owner_root = gather_graph_owner_root_context(
    graph_index_by_occurrence,
    owner_root_by_graph,
    z_after_root,
    is_root_occurrence,
)
r_same, r_owner, deduplicated = deduplicate_identical_feedback_sources(
    r_same, has_same_root, r_owner, has_owner_root
)

features = cat([
    b, z_prev, h_context, r_same, r_owner, scope,
    has_graph_context, has_same_root, has_owner_root, deduplicated,
], dim=-1)

candidate = tanh(context_candidate(features))
gate = sigmoid(context_gate(features))
delta_next = dropout(gate * candidate)
z_next = layer_norm(b + delta_next)
```

Compute all feedback gather tensors before applying the context network. The operations are batched over active occurrences; they do not require Python loops over individual entities or graphs. For memory-limited training, process occurrence chunks only after all feedback source states have been snapshotted, so chunk order cannot change the result.

## 7. First-round and second-round behavior

- After round 1, Stage 8 combines updated canonical states with round-1 contextual states, graph vectors, and ROOT feedback. Its output becomes the input occurrence state for round 2.
- After the final configured round, use Stage 8's `z_next` as the final context-aware occurrence representation for the decoder. Use that round's `b_entity_new` as the final canonical representation. Do not decode from the pre-Stage-8 occurrence snapshot unless an explicitly named ablation requests it.
- Reuse Stage 8 parameters across both rounds by default. Round 2 differs because its inputs contain information propagated during round 1.

Do not apply Stage 1 initialization again between rounds; Stage 8 is the recurrent state construction path. Do not detach the first round before the second during end-to-end training.

## 8. Missing/empty context behavior

- Missing same-entity ROOT occurrence: use zero vector and `has_same_entity_root=0`.
- Missing graph owner ROOT occurrence: use zero vector and `has_owner_root=0`; invalid for an MVP graph that requires a ROOT owner, otherwise follows the configured data policy.
- Atomic ROOT entity with no owned graph: `h_graph_context=0`, flag 0.
- Internal graph with no member occurrences: no occurrence rows to update; preserve its graph state for the next state boundary.
- Empty graph context for a valid occurrence: zero vector and flag 0; do not invent a graph identity embedding.
- No occurrences in a minibatch: return the empty occurrence tensors and canonical states unchanged.

## 9. Gradient flow and leakage constraints

The final prediction loss must flow through the context gate/candidate, the canonical states, Stage 5 graph/entity fusion, Stage 6 ROOT states, Stage 7 synchronization, and previous-round occurrence encoders. Do not detach graph states, root feedback, or canonical states during training.

Only allowed training-graph messages may influence `z_after_root`, `h_graph_fused`, and therefore the next-round adjustments. Held-out target edges and generated reverse channels must remain absent from every earlier stage. Stage 8 itself reads no target relation labels.

## 10. Acceptance criteria

Stage 8 is correct when:

1. Every occurrence state is rebuilt from one canonical state plus a computed contextual adjustment.
2. No trainable parameter row is indexed by GraphNode or `(entity, graph)` occurrence.
3. Same-entity ROOT context and graph-owner ROOT context reach internal occurrences through explicit identity-aligned inputs and masks.
4. An occurrence's containing graph state is included where available.
5. ROOT state is not fed into itself a second time as same-entity feedback.
6. Identical ROOT feedback sources are deduplicated.
7. The owner is not inserted as an internal member and no synthetic KG edge is created.
8. All mappings use explicit masks and remain aligned to the original occurrence IDs.
9. Stage 8 is synchronous and independent of node iteration order.
10. Gradients flow across both propagation rounds.

## 11. Handoff to the next round and decoder

For a second propagation round, Stage 8 outputs:

```text
z_next: [N_occurrence, 256]
delta_next: [N_occurrence, 256]
b_entity_new: [N_entity, 256]
h_graph_fused: [N_internal_graph, 256]
```

Use `z_next` as the occurrence input to the next round's Stage 2 and Stage 3. After the final configured round, pass `b_entity_new` and `z_next` to Stage 9; resolve each candidate to its required occurrence context there.
