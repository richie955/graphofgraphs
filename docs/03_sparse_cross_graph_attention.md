# Stage 3 — Sparse Cross-Graph Relation Attention

**Purpose:** Let information travel across graph contexts only where the GoG data contains an explicit relation between exact `GraphNode` occurrences.

**Inputs:** Stage 2 occurrence states and explicit cross-context relation instances  
**Output:** Context-updated occurrence states  
**Default:** 4 attention heads, hidden size 256, 64 dimensions per head

---

## 1. Responsibility and scope

Stage 3 handles every recorded relation whose endpoint occurrences belong to different graph contexts. This includes:

- ROOT ↔ internal graph relations.
- Internal graph ↔ different internal graph relations.

It excludes:

- Same-internal-graph edges, handled by Stage 2.
- ROOT ↔ ROOT edges, handled by Stage 6.
- Ownership associations between an internal graph and its associated entity. Ownership is not a relation.

The stage is **sparse**: it computes attention only for explicit cross-context relation instances. It must not compare every node in one graph with every node in another graph. Both relation endpoints are updated through computational message channels, but no new biological fact is added to the KG.

## 2. Edge classification and identity

For each stored directed relation:

\[
e=(u,r,v),
\]

where `u` and `v` are exact `GraphNode` occurrences, classify it using their `graph_id` values:

\[
\operatorname{cross}(e) \iff G(u)\ne G(v).
\]

Stage 3 receives only edges satisfying this condition. The two endpoint graph IDs also determine a shared **scope class** used as an attention feature:

- `ROOT_TO_INTERNAL`
- `INTERNAL_TO_ROOT`
- `INTERNAL_TO_INTERNAL`

The scope class is a small shared categorical embedding. Do not learn a separate vector per graph ID. The endpoint occurrence states themselves carry the graph-specific context accumulated so far.

Every edge must retain its original occurrence IDs, relation ID, and graph-scope class through preprocessing and batching. Do not remap an edge to canonical entity IDs alone; that would erase which occurrence the fact connects.

## 3. Inputs and outputs

| Input | Shape | Meaning |
|---|---:|---|
| `z_local` | `[N_occurrence, 256]` | Stage 2 occurrence states |
| `cross_edge_index` | `[2, E_cross]` | Exact occurrence indices `[source, target]` |
| `cross_edge_type` | `[E_cross]` | Shared relation vocabulary IDs |
| `source_graph_index` | `[E_cross]` | Graph ID of each source occurrence |
| `target_graph_index` | `[E_cross]` | Graph ID of each target occurrence |
| `relation_vectors` | `[R, 256]` | Shared trainable relation vectors or references to the shared table |
| `scope_id` | `[E_cross]` | ROOT/internal endpoint-scope class |
| `z_cross` | `[N_occurrence, 256]` | Output states; rows without cross messages remain unchanged |

All graph and relation index values must be validated before the forward pass. `cross_edge_index` is sparse edge data, not a dense adjacency matrix.

## 4. Directed facts and computational directions

For every stored edge (u\xrightarrow{r}v), Stage 3 builds two message channels:

1. **Forward channel:** (u\rightarrow v).
2. **Reverse computational channel:** (v\rightarrow u).

Each channel carries the original relation ID (r), an explicit direction ID, and the endpoint scope class. Separate learned direction features/parameters let the model distinguish the two channels. The reverse channel does not assert that the biomedical relation is semantically reversible.

If the input dataset separately stores an inverse relation as its own semantic fact, preprocessing must follow one consistent convention. Either preserve both stored facts and do not synthesize a second reverse channel for them, or keep one directed fact and generate its computational reverse. Never count the same channel twice.

## 5. Four-head relation- and endpoint-conditioned attention

Let (H=4) heads, (d=256), and (d_h=d/H=64). For one message channel (u\rightarrow v), define:

- (z_u,z_v\in\mathbb R^{256}): states of the exact sender and receiver occurrences.
- (e_r\in\mathbb R^{256}): shared embedding for relation (r).
- (s_{u,v}\in\mathbb R^{d_s}): scope embedding for the endpoint graph-context class plus message direction.

For each head (h\in\{1,\ldots,H\}), calculate:

\[
q_{v,h}=W^Q_hz_v,
\]

\[
k_{u,r,s,h}=W^K_hz_u+U^K_he_r+C^K_hs_{u,v},
\]

\[
v_{u,r,s,h}=W^V_hz_u+U^V_he_r+C^V_hs_{u,v},
\]

where (W^Q_h,W^K_h,W^V_h\in\mathbb R^{d_h\times d}), (U^K_h,U^V_h\in\mathbb R^{d_h\times d}), and (C^K_h,C^V_h) project scope/direction features into head space.

The unnormalized score for the channel is:

\[
\ell_{u\rightarrow v,h}
=\frac{q_{v,h}^{\top}k_{u,r,s,h}}{\sqrt{d_h}}
+\beta_{r,h}+\gamma_{s,h},
\]

where β and γ are learned relation- and scope/direction-conditioned biases. Relation embeddings are shared with the model's relation vocabulary; attention projection parameters are shared across all cross-context edges and graph pairs.

### Attention normalization

For each receiver occurrence (v) and each head (h), normalize over the incoming cross-context message channels targeting (v):

\[
\boxed{\alpha_{u\rightarrow v,h}
=\operatorname{softmax}_{(u,r,d)\in\mathcal N_{cross}(v)}
(\ell_{u\rightarrow v,h})}
\]

This is an edge-wise segmented softmax grouped by receiver occurrence and head. It is **not** a softmax over all nodes in a graph or over all graph pairs.

Aggregate values per receiver:

\[
a_{v,h}=\sum_{(u,r,d)\in\mathcal N_{cross}(v)}
\alpha_{u\rightarrow v,h}v_{u,r,s,h}.
\]

Concatenate and project the heads:

\[
a_v=W^O[a_{v,1}\Vert a_{v,2}\Vert a_{v,3}\Vert a_{v,4}],
\qquad W^O\in\mathbb R^{256\times256}.
\]

The same equations apply to forward and reverse computational channels; their direction/scope features allow the learned projections and biases to treat them differently.

## 6. Gated residual occurrence update

Only occurrences that receive at least one cross-context message are updated by this module. Let (M_v=1) for an active receiver and (M_v=0) otherwise. For active receivers, compute a feature-wise gate:

\[
g_v=\sigma\left(W_g[z_v\Vert a_v]+b_g\right),
\qquad W_g\in\mathbb R^{256\times512},\quad b_g\in\mathbb R^{256}.
\]

Then apply a gated residual update:

\[
\boxed{z_v^{cross}=\operatorname{LayerNorm}\left(z_v+M_v\,\operatorname{Dropout}(g_v\odot a_v)\right)}.
\]

Implementation convention for inactive rows: preserve `z_local` exactly rather than applying LayerNorm to an unchanged value. For active rows, use the learned LayerNorm parameters. This keeps a graph with no cross-context edges from drifting in this stage.

The gate controls how much information from cross-context neighbors is added. The residual preserves the receiver's existing identity/context state. Do not replace an occurrence state with its neighbor aggregate.

## 7. Sparse implementation outline

The model constructs two channel rows per eligible stored edge, unless the data contract indicates an explicit inverse fact that has already been materialized. Each channel row contains sender occurrence, receiver occurrence, relation ID, direction ID, and scope ID.

```python
# z_local is a fixed input snapshot for this entire stage.
channels = make_forward_and_reverse_channels(
    cross_edge_index,
    cross_edge_type,
    scope_id,
)
sender, receiver, relation_id, direction_id, scope_id = channels

# Per-edge, per-head projections; no all-pairs node comparison.
q = project_query(z_local[receiver])
k, value = project_key_value(
    z_local[sender],
    relation_vectors[relation_id],
    direction_id,
    scope_id,
)
score = (q * k).sum(-1) / sqrt(head_dim)
score += relation_bias[relation_id] + scope_direction_bias[scope_id, direction_id]

# Numerically stable segmented softmax by receiver and head.
alpha = segmented_softmax(score, receiver, num_nodes=N_occurrence)
agg_heads = segment_sum(alpha[..., None] * value, receiver, N_occurrence)
agg = output_projection(agg_heads.reshape(N_occurrence, hidden_dim))

active = cross_in_degree > 0
gate = sigmoid(gate_linear(cat([z_local, agg], dim=-1)))
updated_active = layer_norm(z_local[active] + dropout(gate[active] * agg[active]))
z_cross = z_local.index_copy(0, active_indices, updated_active)
```

Use sparse segmented softmax/sum kernels (`scatter`, `index_add`, CSR segment ops, or an equivalent graph library operation). Do not materialize an `[N_occurrence, N_occurrence]` attention matrix. Storage and compute should scale with cross-edge channels, not all possible node pairs.

For mini-batches, attention normalization is over the cross-context incoming channels included in that sampled subgraph. Evaluation must document whether it uses the full available training neighborhood or a fixed sampled neighborhood, and must keep the same leakage rules.

## 8. Data validation and leakage controls

Before invoking the attention module, assert for every edge:

```text
source_graph_index != target_graph_index
source_node_id resolves to the stated source graph
 target_node_id resolves to the stated target graph
```

Use exact ID joins when validating both endpoints.

Additional requirements:

- Apply train/validation/test masks before constructing cross-context channels.
- A held-out target fact must be absent in both forward and generated reverse channels.
- Do not copy generic ROOT facts to internal contexts.
- Do not add edges based only on shared entity identity or ownership.
- Keep duplicate-edge policy explicit; duplicate source facts must not be accidentally repeated by batching.

## 9. Edge cases

| Case | Behavior |
|---|---|
| No cross-context edges | Return `z_local` unchanged |
| One incoming cross message | Softmax weight is 1 for each head |
| Multiple relations from same sender to receiver | Keep distinct channels if distinct stored facts; each has its own relation conditioning |
| ROOT ↔ internal relation | Process the exact endpoint occurrences here |
| Internal ↔ different internal graph relation | Process the exact endpoint occurrences here |
| ROOT ↔ ROOT relation | Exclude; Stage 6 |
| Same-internal-graph relation | Exclude; Stage 2 |
| Entity occurs in several graphs | Do not broadcast a message to all occurrences; update only the endpoints named by this relation |
| Ownership association | Do not process as an edge |
| Explicit inverse relation also present | Follow the one documented inverse-edge convention; avoid duplicated computational channels |
| High-degree receiver | Segment softmax over its actual incoming channels; use configured sparse sampling if needed |

## 10. Trainable parameters and sharing

Trainable Stage 3 components include:

- Four-head query, key, value, and output projections.
- Relation-conditioning projections or biases.
- Shared scope/direction embeddings and biases.
- Gate projection and LayerNorm.
- Dropout is a configuration value, not a parameter.

Parameters are shared over all graph pairs and endpoint occurrences. There is no per-edge, per-pair, or per-occurrence parameter. Relation embeddings may be shared with other stages; record that sharing in the model configuration so the optimizer does not create duplicate tables.

## 11. Gradient flow

The final link-prediction loss must backpropagate through attention weights, values, the gate, Stage 2 states, relation embeddings, and Stage 1 canonical/feature projections. Do not detach cross-edge messages or attention outputs.

## 12. Acceptance criteria

Stage 3 is implemented correctly when:

1. Only explicit relations with endpoints in different graph contexts enter the module.
2. ROOT-to-ROOT and same-internal-graph edges are excluded.
3. Both endpoints of a stored directed cross-context fact can receive messages through distinct computational directions.
4. Relation ID, message direction, and endpoint graph-scope class condition the attention/message calculation.
5. Attention normalization is segmented over incoming edge channels per receiver and head.
6. No dense all-pairs cross-graph attention is materialized.
7. Only named endpoint occurrences are updated; same-entity occurrences elsewhere are not implicitly broadcast to.
8. No ownership edge, inferred edge, or copied fact is introduced.
9. Held-out facts are absent from both message directions.
10. Empty neighborhoods preserve input rows exactly; active rows use a gated residual update.

## 13. Handoff to Stage 4

Stage 4 receives:

```text
z_cross: [N_occurrence, 256]
internal_graph_index_by_occurrence: [N_occurrence]
internal_graph_membership: sparse/indexed mapping of internal graph -> occurrence rows
previous_internal_graph_state: [N_internal_graph, 256]  # if initialized/maintained
```

Stage 4 attention-pools updated internal member states into one representation per internal graph. ROOT remains outside internal graph pooling.
