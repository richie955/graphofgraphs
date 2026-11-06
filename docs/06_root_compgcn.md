# Stage 6 — ROOT-Level CompGCN

**Purpose:** Propagate information among ROOT occurrences through recorded ROOT-to-ROOT KG relations after graph-valued owners have incorporated their internal graph information.

**Inputs:** Stage 5 occurrence states and ROOT-to-ROOT relation instances  
**Output:** Updated ROOT occurrence states; all internal occurrence rows pass through unchanged  
**Default:** 256D, 8 basis matrices per direction, dropout 0.2

---

## 1. Responsibility and boundary

Stage 6 processes **ROOT-to-ROOT relation facts only**. Both endpoint `GraphNode` occurrences must belong to ROOT. This includes ordinary ROOT entities that are atomic and ROOT entities that own internal graphs. A graph-valued ROOT occurrence enters this stage with its Stage 5 upward fusion already applied.

Stage 6 does not process:

- ROOT ↔ internal graph relations. Stage 3 processes these explicit cross-context relations.
- Relations between different internal graphs. Stage 3 processes these.
- Same-internal-graph relations. Stage 2 processes these.
- Owner-to-graph association. Ownership is not a KG edge.

Every recorded ROOT edge is included once in the input fact list. The module generates computational forward and reverse message channels for it. Do not add synthetic reverse edges to the ROOT fact list and then generate another reverse channel.

## 2. Inputs and outputs

| Input | Shape | Meaning |
|---|---:|---|
| `z_owner_all` | `[N_occurrence, 256]` | Stage 5 state tensor with graph-fused owner ROOT rows |
| `root_occurrence_indices` | `[N_root,]` | Global occurrence rows belonging to ROOT |
| `root_edge_index` | `[2, E_root]` | Source/target indices local to the ROOT occurrence array |
| `root_edge_type` | `[E_root]` | Relation IDs from the shared vocabulary |
| `relation_vectors` | `[R, 256]` | Shared relation embeddings |
| `z_root_updated` | `[N_occurrence, 256]` | Full tensor with only ROOT occurrence rows updated |

Keep the distinction between global occurrence indices and ROOT-local edge indices explicit. Convert ROOT-local endpoints to global occurrence rows only when gathering/writing the final tensor.

## 3. Edge and direction construction

For each stored ROOT fact (u\xrightarrow{r}v), construct two computational channels:

- Forward (u\rightarrow v), with direction ID `FORWARD`.
- Reverse (v\rightarrow u), with direction ID `REVERSE`.

The reverse channel supports bidirectional representation propagation. It does not claim that a directed biomedical predicate is semantically symmetric. If inverse relation facts are separately stored as semantic facts, use a single consistent convention and avoid duplicate messages.

The edge preprocessing invariant is:

```text
source.graph_id == ROOT
 target.graph_id == ROOT
```

Use exact endpoint occurrence indices. Do not replace endpoints with only canonical entity IDs.

## 4. Relation transforms with basis decomposition

Let (e_r\in\mathbb R^{256}) be the shared relation vector. Compose it with the sender state using elementwise multiplication:

\[
\phi(z_u,e_r)=z_u\odot e_r.
\]

Build direction-specific effective transforms from shared ROOT-specific bases:

\[
\boxed{W_{r,d}^{root}=\sum_{k=1}^{B}a_{r,d,k}^{root}V_{d,k}^{root}},
\qquad d\in\{forward,reverse\}.
\]

Use (B=8) by default. Each basis (V_{d,k}^{root}\in\mathbb R^{256\times256}); coefficients (a_{r,d,k}^{root}\) are learned per relation and direction. The ROOT bases and coefficients are separate from Stage 2's local internal-graph bases and coefficients. Relation vectors may be shared across stages.

For a channel from sender (u) to receiver (v):

\[
\boxed{m_{u\rightarrow v}^{root}=W_{r,d}^{root}(z_u\odot e_r)}.
\]

No full matrix is allocated per relation instance or edge.

## 5. Aggregate and update ROOT occurrences

For each ROOT occurrence (v), collect incoming forward and reverse computational channels in ℕroot(v). Mean-aggregate by channel count:

\[
a_v^{root}=\frac{1}{c_v}\sum_{(u,r,d)\in\mathcal N_{root}(v)}m_{u\rightarrow v}^{root},
\qquad c_v=\max(1,|\mathcal N_{root}(v)|).
\]

Apply the ROOT-specific self transform, ReLU, dropout, residual connection, and LayerNorm:

\[
q_v=\operatorname{ReLU}(W_{self}^{root}z_v^{in}+a_v^{root}),
\]

\[
\boxed{z_v^{root}=\operatorname{LayerNorm}\left(z_v^{in}+\operatorname{Dropout}(q_v)\right)}.
\]

Updates are synchronous: all ROOT messages read the same `z_owner_all` snapshot. A ROOT node with no ROOT edges still receives the self-loop update. This allows its own Stage 5 fusion state to continue through the ROOT encoder.

## 6. Sparse implementation outline

```python
z_root = z_owner_all[root_occurrence_indices]       # [N_root, d]
src, dst = root_edge_index                           # ROOT-local indices
rel = root_edge_type

# Expand each stored ROOT fact once into two computational channels.
sender = cat([src, dst])
receiver = cat([dst, src])
rel_id = cat([rel, rel])
direction = cat([FORWARD_ids, REVERSE_ids])

composed = z_root[sender] * relation_vectors[rel_id]
messages = root_basis_transform(composed, rel_id, direction)
agg = segment_sum(messages, receiver, num_segments=N_root)
degree = segment_count(receiver, num_segments=N_root).clamp_min(1)
agg = agg / degree[:, None]

update = relu(root_self_linear(z_root) + agg)
z_root_new = root_layer_norm(z_root + dropout(update))

# Write only ROOT rows into the full occurrence tensor.
z_root_updated = z_owner_all.index_copy(
    0, root_occurrence_indices, z_root_new
)
```

Use sparse channel arrays and grouped relation transforms or an equivalent fused sparse kernel. Do not build dense ROOT adjacency or materialize a transformation matrix per edge. If a batch has no ROOT nodes, return the input unchanged.

## 7. Parameters

For (R) relation types, (B=8), and (d=256):

| Component | Parameter count |
|---|---:|
| Directional bases: (2Bd^2) | 1,048,576 |
| Relation-direction coefficients: (2RB) | (16R) |
| ROOT self transform: (d^2) | 65,536 |
| LayerNorm scale and bias: (2d) | 512 |
| **Total Stage 6** | **1,114,624 + 16R** |

These parameters are shared across ROOT occurrences and edges. Reuse the same Stage 6 module across the two propagation rounds by default. Keep ROOT relation transforms distinct from internal local transforms.

## 8. Leakage and preprocessing validation

- Apply train/validation/test masks before building `root_edge_index`.
- Generate reverse computational channels only from the permitted training facts.
- Verify both endpoints resolve to ROOT occurrences.
- Reject cross-context edges in `root_edge_index`; those belong to Stage 3.
- Do not include ownership associations or copied context edges.
- Keep duplicate semantic fact policy explicit. Repeated distinct facts may contribute separate messages; accidental duplicate rows must be removed or reported.

## 9. Edge cases

| Case | Behavior |
|---|---|
| Atomic ROOT entity | Participates normally |
| Graph-valued ROOT entity | Uses Stage 5 fused state as its input |
| ROOT node with no ROOT edges | Self-loop, residual, activation, dropout, LayerNorm |
| No ROOT edges in batch | Self-only updates for ROOT nodes |
| ROOT ↔ internal edge | Exclude; Stage 3 |
| Internal ↔ internal edge | Exclude; Stage 2 if same graph, otherwise Stage 3 |
| Same entity has internal occurrences | Do not update those rows here; Stage 7 synchronizes identities |
| Recorded directed edge | Generate one forward and one reverse computational channel |
| Explicit inverse fact already stored | Follow one documented convention; avoid double-counting |
| Ownership mapping | Never insert into the edge list |

## 10. Gradient flow and acceptance criteria

The link-prediction loss must backpropagate through ROOT attention-free relational messages, relation vectors, ROOT bases/coefficients, self-loop, normalization, Stage 5 owner fusion, and earlier stages. Do not detach `z_owner_all` or `z_root_new`.

Stage 6 is correct when:

1. Only ROOT-to-ROOT facts enter its relation channels.
2. Each stored fact is represented once in input and contributes one forward plus one reverse computational channel.
3. Relation and direction condition message transformations.
4. Effective transforms use basis decomposition and are shared across edges.
5. Incoming messages use a documented channel-count mean.
6. Updates are synchronous with a self path and residual.
7. Only ROOT occurrence rows are replaced in the full tensor.
8. Held-out target edges and their reverse channels are absent.

## 11. Handoff to Stage 7

Stage 7 receives the complete occurrence tensor after ROOT updates. ROOT rows reflect Stages 3, 5, and 6; internal rows reflect Stages 2 and 3. It groups occurrence messages by canonical `entity_id`, aggregates them once per entity, and updates one canonical state per global entity.
