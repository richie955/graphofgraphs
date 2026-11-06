# Stage 5 — Bidirectional Graph–Entity Fusion

**Purpose:** Exchange information between each internal graph representation and the global entity associated with that graph, while preserving one canonical identity per entity and one computed state per occurrence.

**Inputs:** Stage 3 occurrence states, Stage 4 internal graph states, owner mappings  
**Outputs:** Fused owner ROOT occurrence states and fused internal graph states  
**Default:** Separate shared gated residual modules for graph-to-entity and entity-to-graph; 256D states; dropout 0.2

---

## 1. Responsibility and boundary

For an internal graph (G_v) associated with entity (v), Stage 5 exchanges information in two directions:

1. **Upward:** the graph representation informs the associated entity's ROOT occurrence. This lets internal structure affect ROOT-level relation prediction in Stage 6.
2. **Downward:** the updated owner occurrence informs the internal graph representation. Stage 8 uses the resulting graph and owner context to recompute contextual occurrence states for the next round.

Ownership is association metadata. Stage 5 does not create an edge, a relation type, a new entity, or a trainable embedding for the owner/graph pair. It does not directly update the canonical entity vector (b_v); Stage 7 synchronizes occurrence information into that one canonical identity.

## 2. Data model and owner resolution

Each internal graph has an `owner_entity_id`. Resolve that ID to the unique ROOT `GraphNode` occurrence of the same canonical entity:

| Internal graph | Owner entity | Associated ROOT occurrence |
|---|---|---|
| (G_1) | (v_1) | `GraphNode(entity_id=v1, graph_id=ROOT)` |
| (G_2) | (v_2) | `GraphNode(entity_id=v2, graph_id=ROOT)` |

The root occurrence is the contextual state that Stage 5 updates for Stage 6. Do not automatically add the owner entity as a member of its own internal graph. Graph membership remains defined only by actual `GraphNode` rows in that graph.

For MVP 1, use one internal graph per graph-valued owner entity. Validate that each internal graph resolves to exactly one owner entity and exactly one owner ROOT occurrence, and that an owner is not associated with multiple internal graphs. If the dataset intentionally allows a missing ROOT occurrence or multiple owned graphs, stop and define a model extension; do not silently choose an arbitrary occurrence.

## 3. Inputs and outputs

| Input | Shape | Meaning |
|---|---:|---|
| `z_cross` | `[N_occurrence, 256]` | Occurrence states after Stage 3 |
| `h_graph` | `[N_internal_graph, 256]` | Graph representations after Stage 4 |
| `owner_entity_index` | `[N_internal_graph]` | Canonical entity row associated with each graph |
| `owner_root_occurrence_index` | `[N_internal_graph]` | Exact ROOT occurrence row to fuse |
| `graph_nonempty_mask` | `[N_internal_graph]` | Whether the graph has at least one member occurrence |
| `z_owner_fused` | `[N_occurrence, 256]` | Input occurrence tensor with owner ROOT rows replaced by upward-fused states |
| `h_graph_fused` | `[N_internal_graph, 256]` | Graph states after owner-to-graph fusion |
| `owner_feedback` | `[N_internal_graph, 256]` | Owner context exposed to Stage 8 |

`owner_entity_index` is an identity mapping for validation and traceability; it is not an embedding lookup specific to a graph. Only `owner_root_occurrence_index` selects the contextual ROOT row to update here.

## 4. Upward fusion: internal graph to owner ROOT occurrence

For graph (G_v), let:

- (z_{v,ROOT}^{cross}\in\mathbb R^{256}): current owner ROOT occurrence state from Stage 3.
- (h_{G_v}^{new}\in\mathbb R^{256}): internal graph state from Stage 4.

Concatenate the two inputs:

\[
x_v^{up}=[z_{v,ROOT}^{cross}\Vert h_{G_v}^{new}]\in\mathbb R^{512}.
\]

Compute a candidate and feature-wise gate with shared upward parameters:

\[
c_v^{up}=\tanh(W_c^{up}x_v^{up}+b_c^{up}),
\]

\[
g_v^{up}=\sigma(W_g^{up}x_v^{up}+b_g^{up}),
\]

where both weight matrices are (256\times512) and both biases are 256D. Update the owner ROOT occurrence:

\[
\boxed{z_{v,ROOT}^{up}=\operatorname{LayerNorm}\left(z_{v,ROOT}^{cross}+\operatorname{Dropout}(g_v^{up}\odot c_v^{up})\right)}.
\]

This state is used by Stage 6. It contains the previous owner context, cross-context messages from Stage 3, and learned information from its owned graph. It is still an occurrence state, not a new canonical entity identity.

## 5. Downward fusion: owner ROOT occurrence to graph state

Use the upward-fused owner occurrence so the downward path can carry any owner update that incorporated the graph summary. For the same pair:

\[
x_v^{down}=[h_{G_v}^{new}\Vert z_{v,ROOT}^{up}]\in\mathbb R^{512}.
\]

Calculate an independent candidate and gate with shared downward parameters:

\[
c_v^{down}=\tanh(W_c^{down}x_v^{down}+b_c^{down}),
\]

\[
g_v^{down}=\sigma(W_g^{down}x_v^{down}+b_g^{down}),
\]

\[
\boxed{h_{G_v}^{fused}=\operatorname{LayerNorm}\left(h_{G_v}^{new}+\operatorname{Dropout}(g_v^{down}\odot c_v^{down})\right)}.
\]

The upward and downward modules have separate parameters because they transfer information between different representations in different directions. Both modules are shared across all graph-owner pairs.

Expose:

\[
owner\_feedback_v=z_{v,ROOT}^{up}
\]

along with (h_{G_v}^{fused}) to Stage 8. Stage 8 uses the updated canonical identity, graph context, and owner feedback to compute next-round occurrence adjustments. Stage 5 does not broadcast a raw graph summary into every member row.

## 6. Vectorized implementation outline

Owner mappings must be unique so that `index_copy` does not have write collisions.

```python
# Inputs are fixed snapshots for this stage.
owner_idx = owner_root_occurrence_index[graph_nonempty_mask]
graph_idx = nonempty_graph_indices
z_owner = z_cross[owner_idx]                 # [Q, 256]
h_owner = h_graph[graph_idx]                  # [Q, 256]

# Upward: graph -> owner ROOT occurrence.
up_input = torch.cat([z_owner, h_owner], dim=-1)  # [Q, 512]
up_candidate = torch.tanh(up_candidate_linear(up_input))
up_gate = torch.sigmoid(up_gate_linear(up_input))
z_owner_up = layer_norm_up(
    z_owner + dropout(up_gate * up_candidate)
)

# Downward: updated owner -> graph context.
down_input = torch.cat([h_owner, z_owner_up], dim=-1)
down_candidate = torch.tanh(down_candidate_linear(down_input))
down_gate = torch.sigmoid(down_gate_linear(down_input))
h_owner_fused = layer_norm_down(
    h_owner + dropout(down_gate * down_candidate)
)

z_owner_all = z_cross.index_copy(0, owner_idx, z_owner_up)
h_graph_fused = h_graph.index_copy(0, graph_idx, h_owner_fused)
owner_feedback = make_graph_aligned_feedback(owner_idx, z_owner_up)
```

The output `z_owner_all` has the same shape as `z_cross`; only owner ROOT occurrence rows are replaced. The graph output retains one row per internal graph. The implementation must preserve graph-to-owner row alignment when filtering empty graphs.

## 7. Empty graphs and missing owners

- **Non-empty graph:** run both fusion directions.
- **Empty graph:** Stage 4 preserves its previous graph context. Skip Stage 5 fusion for that graph in MVP 1 so an empty graph cannot change its owner through projection biases alone. Pass the owner context through unchanged and mark the graph as inactive for Stage 8 graph-member feedback.
- **Missing owner mapping or missing ROOT occurrence:** preprocessing/configuration error. Do not pick another occurrence or manufacture an owner relation.
- **Atomic entity without an internal graph:** no Stage 5 operation.

## 8. Parameter count and sharing

Each direction uses a candidate linear layer, gate linear layer, and LayerNorm:

| Component per direction | Parameters at (d=256) |
|---|---:|
| Candidate `Linear(512, 256)` | 131,328 |
| Gate `Linear(512, 256)` | 131,328 |
| LayerNorm | 512 |
| **Total per direction** | **263,168** |

With independent upward and downward modules, Stage 5 has **526,336** parameters at (d=256). Parameters are shared across every graph-owner pair and reused across the two propagation rounds. There are no owner-specific or graph-specific fusion parameters.

## 9. Gradient flow and update ordering

The downstream prediction loss must backpropagate through both fusion directions, Stage 4 graph readout, Stage 3 occurrence states, Stage 2 local propagation, and Stage 1 canonical/feature parameters. Do not detach (h_G^{new}), (z_{v,ROOT}^{up}), or (h_G^{fused}).

Within one round, use this ordering:

1. Read the Stage 3 owner ROOT occurrence and Stage 4 graph state snapshots.
2. Compute all upward owner fusions.
3. Compute all downward graph fusions from the upward owner outputs.
4. Replace owner ROOT rows for Stage 6 and pass fused graph state/owner feedback to Stage 8.
5. Continue to Stage 6 using the updated ROOT occurrence tensor.

Do not update the canonical entity table in place here. Stage 7 performs global synchronization after the ROOT relational stage.

## 10. Leakage and graph semantics

- Graph ownership is not inserted in `edge_index` and is not scored as a relation.
- Only graph structure available in the training message-passing graph can affect the pooled graph state.
- Held-out relation facts remain excluded from Stages 2 and 3, so they cannot influence the Stage 4 graph summary or this fusion.
- Owner selection uses identity and graph association metadata, not validation/test relation labels.
- No graph summary is copied to entities or occurrences that are not in the owner mapping.

## 11. Acceptance criteria

Stage 5 is implemented correctly when:

1. Every non-empty internal graph is paired with its intended global owner and unique ROOT occurrence.
2. The graph state updates the owner ROOT occurrence through a shared gated residual path.
3. The upward-fused owner context updates the graph state through a separate shared gated residual path.
4. Atomic entities bypass this stage.
5. Empty graphs skip fusion and preserve owner/graph states according to the documented policy.
6. No owner, graph, or occurrence-specific trainable identity is introduced.
7. The canonical entity state is not directly modified; Stage 7 remains responsible for global identity synchronization.
8. Only the designated owner ROOT occurrence is changed by the upward path.
9. The graph owner mapping does not create or imply a KG edge.
10. Gradients remain connected through both fusion paths.

## 12. Handoff to Stage 6 and Stage 8

Stage 6 receives the occurrence tensor with updated owner ROOT rows:

```text
z_owner_all: [N_occurrence, 256]
root_occurrence_indices: [N_root_occurrence]
root_edge_index: [2, E_root]  # ROOT-to-ROOT facts only
root_edge_type: [E_root]
```

Stage 8 receives the Stage 7 canonical state, the fused graph states, the graph-aligned owner feedback, and occurrence/graph mappings. It computes contextual adjustments and downward feedback for the next round; it must not convert ownership into a relation or allocate occurrence identity parameters.
