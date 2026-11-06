# Stage 2 — Local CompGCN Inside Internal Graphs

**Purpose:** Propagate typed relational information among occurrences connected inside the same internal graph. This is the first structure-aware message-passing stage.

**Inputs:** Stage 1 occurrence states and internal same-graph relation edges  
**Output:** Updated internal occurrence states, still 256D  
**Default:** 8 basis matrices per direction, 2 computational message directions, dropout 0.2

---

## 1. Responsibility and boundary

Stage 2 processes only relation instances whose source and target `GraphNode` occurrences belong to the same **non-ROOT internal graph**. It applies a shared relation-aware CompGCN-style update to those nodes.

It does not:

- Process ROOT-to-ROOT edges (Stage 6).
- Process ROOT-to-internal, internal-to-ROOT, or internal-to-different-internal graph edges (Stage 3).
- Pool member nodes into a graph representation (Stage 4).
- Treat graph ownership as an edge.
- Copy any edge to another graph context.

Relation scope comes from endpoint occurrence graph IDs, not the relation name. A relation called `interacts_with` can appear at different scopes; Stage 2 uses it only for same-internal-graph endpoint pairs.

## 2. Inputs and outputs

| Tensor / mapping | Shape | Meaning |
|---|---:|---|
| `z_in` | `[N_occurrence, d]` | Stage 1 states, (d=256) by default |
| `internal_local_edge_index` | `[2, E_local]` | Source and target indices in the global occurrence tensor, or a documented concatenated internal-local index space |
| `internal_local_edge_type` | `[E_local]` | Shared relation vocabulary IDs |
| `internal_graph_index_by_occurrence` | `[N_occurrence]` | Graph index for endpoint scope validation |
| `relation_vectors` | `[R, d]` | Shared relation embeddings, aligned to relation IDs |
| `z_local` | `[N_occurrence, d]` | Output states; non-internal rows pass through unchanged |

If the implementation remaps nodes to a compact local tensor, it must return an explicit `local_to_global_occurrence` mapping and preserve the same semantics.

## 3. Edge channels and direction semantics

A stored fact is directed:

\[
(u,r,v),
\]

where (u) and (v) are exact occurrence nodes in the same internal graph. For message passing, construct two computational channels:

1. Forward: source (u) sends to target (v).
2. Reverse: target (v) sends to source (u), using a separate reverse-direction transformation.

The reverse channel is an encoder operation. It does not add a reverse biomedical fact to the data. Do not also materialize an inverse edge in `internal_local_edge_index` unless preprocessing has a single documented convention that prevents duplicate messages.

Use the relation embedding for the stored relation ID in both directions, with direction-specific transform parameters. If the dataset explicitly models inverse relations as separate semantic relation types, preserve those IDs as supplied and do not additionally synthesize duplicate channels.

## 4. Relation-aware message construction

Let (z_u) be the source occurrence vector and (e_r) the shared embedding for relation (r). For this MVP, use elementwise multiplicative composition:

\[
\phi(z_u,e_r)=z_u\odot e_r.
\]

Construct separate forward/reverse transform matrices using basis decomposition:

\[
\boxed{W_{r,d}=\sum_{k=1}^{B}a_{r,d,k}V_{d,k}}, \quad d\in\{\mathrm{fwd},\mathrm{rev}\},
\]

where (B=8) by default, (V_{d,k}\in\mathbb R^{d\times d}), and (a_{r,d,k}\) is a learned scalar coefficient. Then the message on edge ((u,r,v)) is:

\[
m^{\mathrm{fwd}}_{u\to v}=W_{r,\mathrm{fwd}}(z_u\odot e_r),
\]

\[
m^{\mathrm{rev}}_{v\to u}=W_{r,\mathrm{rev}}(z_v\odot e_r).
\]

The same basis matrices and coefficients are shared across all internal graphs and all edges. Do not allocate a full matrix per edge. Local CompGCN parameters are separate from Stage 6 ROOT CompGCN parameters so the model can learn different transformations for internal and ROOT structure.

## 5. Aggregation and node update

For node occurrence (v), collect all incoming forward and reverse message channels targeting it. Let ℕ(v) be that multiset of channels, and let (c_v=\max(1,|\mathcal N(v)|)). Mean-aggregate:

\[
a_v=\frac{1}{c_v}\sum_{(u,r,d)\in\mathcal N(v)}m_{u\to v}^{d}.
\]

Use a shared self-loop transform (W_{self}^{local}\in\mathbb R^{d\times d}), ReLU, dropout, residual, and LayerNorm:

\[
q_v=\operatorname{ReLU}(W_{self}^{local}z_v^{in}+a_v),
\]

\[
\boxed{z_v^{local}=\operatorname{LayerNorm}\left(z_v^{in}+\operatorname{Dropout}(q_v)\right)}.
\]

Updates are synchronous. All messages for the layer are computed from the same `z_in` snapshot; do not update one node in place and then use its updated state to compute a neighbor message in the same layer.

An isolated node has (a_v=0) and still receives the self-loop/residual transformation. If a graph has no local edges, every node in that graph follows this deterministic self-only update path.

## 6. Sparse implementation plan

Preprocessing must classify edges once and verify for every local edge:

```text
source.graph_id == target.graph_id
source.graph_id != ROOT
```

Use sparse edge-index tensors. Avoid dense adjacency matrices. Build forward and reverse channel arrays with `src`, `dst`, `relation_id`, and `direction` columns, or equivalent typed tensors.

Efficient implementation outline:

```python
# One fixed input snapshot for the whole layer.
src, dst = local_edge_index
rel = local_edge_type

# Expand each recorded edge into forward and reverse computational channels.
senders = cat([src, dst])
receivers = cat([dst, src])
rel_ids = cat([rel, rel])
directions = cat([FORWARD * E_local, REVERSE * E_local])

composed = z_in[senders] * relation_vectors[rel_ids]
messages = relation_basis_transform(composed, rel_ids, directions)
agg = segment_sum(messages, receivers, num_nodes=N_occurrence)
degree = segment_count(receivers, num_nodes=N_occurrence).clamp_min(1)
agg = agg / degree[:, None]

update = relu(local_self_linear(z_in) + agg)
z_candidate = layer_norm(z_in + dropout(update))
z_local = replace_only_internal_occurrences(z_in, z_candidate)
```

`segment_sum` and `segment_count` may use `index_add`, scatter operations, or a graph library kernel. Do not compute matrices per edge. If relation-grouped kernels are used, grouping is an implementation detail and must not change normalization or edge semantics.

### Basis transform shape

One valid storage layout is:

```text
basis:        [2, B, d, d]
coefficients: [R, 2, B]
```

Effective matrices can be formed on demand:

```python
W = einsum("rdb,dbij->rdij", coefficients, basis)  # [R, 2, d, d]
```

Avoid building `W` repeatedly inside the per-edge loop. Either cache the effective matrices for the forward pass or apply the basis expansion to grouped relation batches without materializing edge-level matrices.

The total local transform parameters are approximately:

\[
2Bd^2 + 2RB + d^2 + 2d,
\]

including the two-direction bases, coefficients, self transform, and LayerNorm scale/bias. With (B=8,d=256), this remains independent of the number of relation **instances** and number of graphs.

## 7. Relation vectors and sharing

`relation_vectors` are aligned to the global relation vocabulary and may be shared with the cross-graph attention and ROOT modules as their stage specifications define. Stage 2 owns its basis matrices, relation coefficients, self-loop transform, and normalization parameters. Stage 2 must not create a second unrelated relation ID vocabulary.

Use a stable mapping from relation names/IDs to contiguous indices and store it with the checkpoint. Unknown relation IDs at inference must follow an explicit configured unknown-relation path; do not index out of bounds or silently map to an arbitrary known relation.

## 8. Validation and leakage boundaries

Before building `internal_local_edge_index`, apply the training/evaluation split. A validation/test target fact must not enter Stage 2. If reverse channels are generated from the training edge list, only training facts get reverse channels.

Preprocessing validation should reject or report:

- Edge endpoints that do not resolve to GraphNodes.
- Edges classified as local when endpoint graph IDs differ.
- ROOT edges accidentally present in the local edge list.
- Duplicate edge instances when the source data does not permit duplicates.
- Synthetic ownership edges or copied facts.

## 9. Gradient flow

The downstream link-prediction loss must backpropagate through:

- Relation vectors (e_r), if trainable/shared.
- Basis matrices (V_{d,k}).
- Relation coefficients (a_{r,d,k}).
- Self-loop matrix and LayerNorm.
- Stage 1 canonical and feature projection parameters through `z_in`.

Do not detach messages or the Stage 2 output. Do not keep persistent trainable states per occurrence between optimizer steps; recompute occurrence states from current canonical states and graph inputs.

## 10. Edge cases

| Case | Behavior |
|---|---|
| One internal node, no edges | Self-only residual update |
| Internal graph with edges | Process all its within-graph edge channels |
| ROOT-to-ROOT edge | Excluded; Stage 6 |
| ROOT-to-internal or internal-to-ROOT edge | Excluded; Stage 3 |
| Edge between two different internal graphs | Excluded; Stage 3 |
| Multiple relation instances between same endpoints | Keep distinct if they are distinct facts; each contributes a channel |
| Self-loop fact | Process according to source data policy; do not add a second implicit edge beyond the module self-loop |
| Empty edge batch | Return valid states using the self-only update path, or return unchanged if the stage configuration explicitly disables self-only updates; use one behavior consistently |
| No internal occurrences in batch | Return input tensor unchanged without error |

## 11. Acceptance criteria

Stage 2 is implemented correctly when:

1. Only same-internal-graph edges enter the local encoder.
2. Every edge endpoint remains the exact GraphNode occurrence specified in the source data.
3. Forward and reverse computational channels use direction-specific parameters without duplicate edge insertion.
4. Messages are relation-conditioned, and transforms use basis decomposition rather than one matrix per edge.
5. Aggregation is sparse and uses a documented incoming-channel normalization.
6. The layer update is synchronous, includes a self path, and handles isolated nodes deterministically.
7. Non-internal occurrence rows are preserved according to the declared output contract.
8. Held-out facts are absent from both forward and reverse channels.
9. Gradients reach Stage 1 and Stage 2 trainable parameters.

## 12. Handoff to Stage 3

Stage 3 receives the Stage 2 occurrence states and only the explicit cross-context relation instances:

```text
z_local: [N_occurrence, 256]
cross_edge_index: [2, E_cross]
cross_edge_type: [E_cross]
source_graph_index / target_graph_index: derived from endpoint occurrences
```

Stage 3 must use sparse relation-/endpoint-conditioned attention on these edges. Stage 2 must not pre-aggregate or copy cross-context facts.
