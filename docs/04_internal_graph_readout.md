# Stage 4 — Internal Graph Attention Readout

**Purpose:** Compress each internal graph's current occurrence states into one graph representation, retaining a learned summary of its structure and cross-context messages.

**Inputs:** Stage 3 occurrence states, internal graph membership, previous internal graph states  
**Output:** One updated 256D graph state per internal graph  
**Default:** Shared single-head additive attention pooling followed by a gated residual update

---

## 1. Responsibility and scope

Stage 4 builds one representation (h_G) for each internal graph (G). Its input nodes are the exact `GraphNode` occurrences whose `graph_id` equals (G), after local propagation (Stage 2) and explicit cross-context attention (Stage 3).

This stage does not:

- Pool the ROOT graph.
- Reprocess relation edges. Relational information is already reflected in Stage 2 and Stage 3 occurrence states.
- Insert the owning entity as a graph member. Ownership remains metadata, not membership or a relation.
- Create a trainable embedding row for every graph ID.

The output is a graph-context representation. Stage 5 will exchange information between this graph vector and its associated graph-valued entity.

## 2. Inputs and outputs

| Input | Shape | Meaning |
|---|---:|---|
| `z_cross` | `[N_occurrence, 256]` | Stage 3 occurrence states |
| `internal_graph_index_by_occurrence` | `[N_occurrence]` | Graph ID/index for each occurrence |
| `internal_graph_member_index` | `[N_internal_graph, variable]` or CSR | Sparse graph-to-occurrence membership |
| `h_graph_prev` | `[N_internal_graph, 256]` | Prior graph context; zero initialized in round 1 |
| `is_internal_graph` | `[N_internal_graph]` | Excludes ROOT |
| `h_graph_new` | `[N_internal_graph, 256]` | Updated graph representations |
| `pool_attention` | per graph, per node | Weights used to construct pooled summaries |

Membership must come from `GraphNode.graph_id`, not from relation edges or owner mappings. Pool occurrences, not distinct canonical entities; each occurrence belongs to its graph context.

## 3. Initial graph context and round-to-round state

Do not create a free trainable graph-ID embedding table. For the first propagation round, initialize each internal graph context as:

\[
h_G^{prev}=\mathbf 0.
\]

The node attention network can therefore score nodes from their current states while the update module learns the initial graph context from the pooled node information. For subsequent rounds, pass the previous round's graph state after the graph/entity feedback stages as `h_graph_prev`.

This makes graph context content-derived and recurrent across rounds without allocating independent graph identity parameters. An empty graph starts at zero and retains its previous state.

## 4. Context-conditioned additive attention

For every occurrence (v\in G), let (z_{v,G}^{cross}\in\mathbb R^{256}) be its current Stage 3 state, and let (h_G^{prev}\in\mathbb R^{256}) be the previous graph context.

Calculate one scalar score per occurrence using a shared additive attention scorer:

\[
\boxed{s_{v,G}=w_a^\top\tanh(W_z z_{v,G}^{cross}+W_h h_G^{prev}+b_a)}.
\]

Dimensions:

- (W_z,W_h\in\mathbb R^{256\times256}).
- (b_a,w_a\in\mathbb R^{256}).
- (s_{v,G}\in\mathbb R).

Normalize over the member occurrences of the same graph only:

\[
\boxed{\alpha_{v,G}=\operatorname{softmax}_{v\in G}(s_{v,G})}.
\]

Then compute the pooled node summary:

\[
\boxed{p_G=\sum_{v\in G}\alpha_{v,G}z_{v,G}^{cross}},
\qquad p_G\in\mathbb R^{256}.
\]

The same attention scorer is used for every internal graph. The graph context (h_G^{prev}) conditions which of its nodes contribute most. For a one-node graph, the node receives attention weight 1.

## 5. Gated residual graph update

The pooled summary (p_G) contains current node information; (h_G^{prev}) carries graph context from initialization or a preceding round. Combine them with shared candidate and gate networks.

First generate a candidate update:

\[
c_G=\tanh\left(W_p[h_G^{prev}\Vert p_G]+b_p\right),
\]

\[
W_p\in\mathbb R^{256\times512},\qquad b_p\in\mathbb R^{256}.
\]

Then calculate a feature-wise gate:

\[
g_G=\sigma\left(W_g[h_G^{prev}\Vert p_G]+b_g\right),
\]

\[
W_g\in\mathbb R^{256\times512},\qquad b_g\in\mathbb R^{256}.
\]

Finally update the graph state:

\[
\boxed{h_G^{new}=\operatorname{LayerNorm}\left(h_G^{prev}+\operatorname{Dropout}(g_G\odot c_G)\right)}.
\]

Dropout is 0.2 by default and configurable. The gate controls how much candidate information is added; the residual path preserves previous graph context. Use the same candidate, gate, and LayerNorm parameters for all internal graphs.

## 6. Complete stage equations

For each non-empty internal graph (G):

\[
\begin{aligned}
s_{v,G}&=w_a^\top\tanh(W_z z_{v,G}^{cross}+W_hh_G^{prev}+b_a),\\
\alpha_{v,G}&=\operatorname{softmax}_{v\in G}(s_{v,G}),\\
p_G&=\sum_{v\in G}\alpha_{v,G}z_{v,G}^{cross},\\
c_G&=\tanh(W_p[h_G^{prev}\Vert p_G]+b_p),\\
g_G&=\sigma(W_g[h_G^{prev}\Vert p_G]+b_g),\\
h_G^{new}&=\operatorname{LayerNorm}(h_G^{prev}+\operatorname{Dropout}(g_G\odot c_G)).
\end{aligned}
\]

All graph updates in a round are computed from the same `z_cross` and `h_graph_prev` snapshots. Do not let an updated graph state affect another graph's pooling in the same stage.

## 7. Sparse implementation outline

Use CSR pointers, graph-segment IDs, or an equivalent ragged-batch representation. Softmax groups by internal graph index; the weighted reduction is also segmented by graph.

```python
# z_cross: [N_occurrence, d]
# member_graph_idx: [N_internal_members]
# member_occ_idx:   [N_internal_members]

z_member = z_cross[member_occ_idx]
h_member_prev = h_graph_prev[member_graph_idx]

hidden = tanh(
    linear_node(z_member)
    + linear_graph(h_member_prev)
    + attention_bias
)
scores = hidden @ attention_vector                 # [N_internal_members]
alpha = segmented_softmax(scores, member_graph_idx)
p_graph = segment_sum(
    alpha[:, None] * z_member,
    member_graph_idx,
    num_segments=N_internal_graph,
)

combined = cat([h_graph_prev, p_graph], dim=-1)     # [N_internal_graph, 512]
candidate = tanh(candidate_linear(combined))        # [N_internal_graph, 256]
gate = sigmoid(gate_linear(combined))               # [N_internal_graph, 256]
h_candidate = layer_norm(
    h_graph_prev + dropout(gate * candidate)
)

# Keep empty graph rows equal to their previous contexts.
h_graph_new = where(nonempty_graph[:, None], h_candidate, h_graph_prev)
```

Do not build a dense `[N_internal_graph, N_occurrence]` membership matrix. Runtime and memory should scale with the number of internal membership rows and node feature dimensions.

The pooling score and weight are useful diagnostics. They may be returned optionally for analysis but must not be treated as target labels or detached in the training computation.

## 8. Parameters

For (d=256), the shared readout contains:

| Component | Parameters |
|---|---:|
| Additive attention (W_z,W_h,b_a,w_a) | 131,584 |
| Candidate projection (W_p,b_p) | 131,328 |
| Gate projection (W_g,b_g) | 131,328 |
| LayerNorm scale and bias | 512 |
| **Total** | **394,752** |

These parameters are shared across all internal graphs and propagation rounds unless a later model revision explicitly changes that decision. They do not scale with the number of graphs. The count assumes no biases on (W_z,W_h) beyond the single (b_a) shown; the implementation should match this convention.

## 9. Edge cases

| Case | Behavior |
|---|---|
| Internal graph with many member occurrences | Attention normalizes over only those occurrences |
| One-member internal graph | Its attention weight is 1; pool equals its occurrence state |
| Empty internal graph | Skip attention and preserve `h_graph_prev` |
| Graph has no text/description | Pool from structural occurrence states normally |
| Member has explicit cross-graph relations | Stage 3 updates its occurrence state before pooling; that information can flow into the graph summary |
| Same canonical entity appears in multiple graphs | Each graph pools its own occurrence row and context |
| Owner entity is not a graph member | Do not insert it into the member set; Stage 5 uses owner mapping separately |
| ROOT graph | Exclude from this module; it follows the ROOT processing path |
| Graph has no previous context in round 1 | Use the zero graph context specified above |
| Graph context updated in earlier round | Use the state supplied by graph/entity feedback as `h_graph_prev` |

## 10. Gradient flow and training

Link-prediction loss must backpropagate through:

- Attention scores and normalized weights.
- Pooled node vectors.
- Candidate and gate networks.
- LayerNorm.
- Stage 3 and earlier occurrence states.

Do not detach `p_G`, `h_G_new`, or attention weights in the encoder. No separate supervision for a graph vector is required; it is learned end-to-end through downstream link scores.

## 11. Acceptance criteria

Stage 4 is implemented correctly when:

1. It produces one 256D representation per internal graph.
2. Pooling uses only that graph's exact GraphNode occurrences.
3. Attention is normalized independently per graph and shared across graphs.
4. The previous graph state participates in both attention scoring and the gated update.
5. Candidate and gate are shared across internal graphs.
6. Empty graphs preserve their prior state; one-node graphs work without special trainable parameters.
7. ROOT is not passed through internal graph pooling.
8. Owner entities are not added to internal membership unless represented by an actual GraphNode in the source schema.
9. The module uses sparse/ragged membership operations, not a dense graph-by-node matrix.
10. Gradients flow through the pooling and graph update into prior stages.

## 12. Handoff to Stage 5

Stage 5 receives:

```text
h_graph_new: [N_internal_graph, 256]
z_cross: [N_occurrence, 256]
owner_entity_index: [N_internal_graph]
owner_root_occurrence_index: [N_internal_graph]  # where such ROOT occurrence exists
```

The owner mapping pairs each internal graph with its associated global entity. Stage 5 must specify the exact upward graph-to-entity and downward entity-to-graph/member fusion paths without converting ownership into a KG relation.
