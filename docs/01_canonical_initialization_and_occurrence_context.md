# Stage 1 — Canonical Initialization and Occurrence Context

**Purpose:** Define the initial vector for each global entity and create the first contextual state for each `GraphNode` occurrence. This stage establishes the identity-sharing contract used by every later stage.

**Architecture source:** `GoG_Full_Pipeline_Overview.md`  
**Hidden dimension:** (d=256) by default  
**Output:** One canonical state per `Entity`, plus one computed state per `GraphNode` occurrence

---

## 1. Responsibility and boundary

Stage 1 prepares identity and occurrence states for the first propagation round. It does not perform relation message passing, cross-graph attention, graph pooling, owner fusion, or ROOT propagation. Those happen in later stages.

The core rule is:

> A real entity has one canonical state shared globally. A graph occurrence has a computed contextual state, never its own free trainable identity vector.

If entity `MEK` appears in ROOT and in the MAPK and PI3K internal graphs, there is one canonical `MEK` state. Stage 1 creates three occurrence rows that all derive from it. Later message passing will make those occurrence states context-specific.

## 2. Inputs and outputs

### Inputs

| Input | Shape / form | Notes |
|---|---|---|
| Stable entity IDs | `[N_entity]` | Global IDs mapped to contiguous model indices |
| Entity type IDs | `[N_entity]` | Shared vocabulary, with an unknown type ID |
| Frozen biomedical text features | `[N_entity, d_text]` | Optional; preprocessing and source must be recorded |
| Optional structured entity features | Dataset-specific | Only include features available consistently at inference |
| GraphNode entity indices | `[N_occurrence]` | Maps each occurrence to its global entity row |
| GraphNode graph indices | `[N_occurrence]` | ROOT or one internal graph |

### Outputs

| Output | Shape | Meaning |
|---|---|---|
| `b_entity` | `[N_entity, 256]` | Canonical initial entity states |
| `delta_occurrence` | `[N_occurrence, 256]` | Initial contextual adjustments; zero at initialization |
| `z_occurrence` | `[N_occurrence, 256]` | Initial occurrence states, derived from canonical state |
| `entity_index_by_occurrence` | `[N_occurrence]` | Stable mapping for later stages |
| `graph_index_by_occurrence` | `[N_occurrence]` | Stable mapping for later stages |

Here (N_{entity}) is the number of global entities and (N_{occurrence}) is the number of materialized `GraphNode` rows in the active dataset/batch.

## 3. Canonical entity initialization

### 3.1 Feature inputs

Use a single canonical representation for each entity. Build it from:

1. A trainable global entity identity vector (e_v).
2. A trainable entity-type embedding (t_{	au(v)}).
3. A projection of frozen biomedical text features when available.
4. Optional structured features only when they are stable, permitted at inference, and defined for the relevant entity class.

Let (x_v^{text}in\mathbb R^{d_{text}}) be the fixed text feature vector. Let (m_v^{text}\in\{0,1\}) indicate whether it is available. Missing text is represented by a zero vector plus the missingness indicator; do not silently substitute a text embedding for another entity.

A concrete default initializer is:

\[
u_v = e_v + t_{\tau(v)} + m_v^{text}W_{text}x_v^{text} + W_{flag}(1-m_v^{text}) + W_{attr}x_v^{attr},
\]

\[
\boxed{b_v^{(0)}=u_v}
\]

Terms for unavailable optional attributes are omitted. The dimensions are:

- (e_v,t_{\tau(v)},u_v,b_v^{(0)}\in\mathbb R^{256}).
- (W_{text}\in\mathbb R^{256\times d_{text}}).
- (W_{flag}\in\mathbb R^{256}), as a learned missing-text vector.
- (W_{attr}\) maps the configured structured feature vector into 256D.

The entity ID table is allowed because it stores **one global parameter row per entity**. Never index it by `node_id`, `graph_id`, or `(entity_id, graph_id)`.

If the project configuration chooses no trainable entity-ID table, use the same formula without (e_v); that is a model ablation and must be named/configured explicitly. Do not silently switch initialization modes between training and inference.

### 3.2 Frozen feature handling

- Keep the biomedical text encoder frozen in MVP 1; store or compute its vectors outside the trainable GoG encoder.
- Train the projection (W_{text}) jointly with the GoG model.
- Record text model/version, text normalization, maximum input length, vector dimension, and missing-feature handling in configuration or a manifest.
- Do not use validation/test relation facts to construct text inputs.
- Any feature normalization statistics must be fitted on training-available entity features only, then reused unchanged.

### 3.3 Entity types and missing values

The type vocabulary is global across ROOT and internal graphs. Unknown/missing types map to a designated `UNK` type embedding. Do not create type embeddings per graph. If an entity has multiple valid types, the data contract must define the multi-type encoding; Stage 1 must not guess between them.

## 4. Initial occurrence state

For an occurrence (o) of entity (v) in graph (G), initialize a contextual adjustment Δ as zero:

\[
\Delta_{v,G}^{(0)}=\mathbf 0.
\]

Then compute:

\[
\boxed{z_{v,G}^{(0)}=\operatorname{LayerNorm}\left(b_v^{(0)}+\Delta_{v,G}^{(0)}\right)}
\]

The canonical base \(b_v^{(0)}\) is the sum of projected identity and feature inputs before occurrence normalization. Apply LayerNorm when constructing each occurrence state, exactly as in the occurrence-state equation. Keep this convention consistent through later stages; Stage 7 must specify whether its updated canonical state is pre-normalization or normalized.

### Why all occurrences can begin equal

Two occurrences of the same entity may have the same vector at the start. That is intentional. Their contexts diverge when relation-specific local and cross-graph messages are computed from different neighborhoods. A free occurrence-specific vector would memorize context IDs and violate the shared-identity design.

If a later model revision chooses to initialize a nonzero context delta from observable graph features, it must be computed by a shared function from those features and documented as part of Stage 1. It still cannot be a trainable lookup table per occurrence.

## 5. Tensor/index implementation contract

Use contiguous integer mappings for efficient tensor access:

```text
entity_id_to_index: global Entity ID -> [0, N_entity)
node_id_to_index:   GraphNode ID -> [0, N_occurrence)
graph_id_to_index:  Graph ID -> [0, N_graph)
entity_index_by_occurrence: LongTensor[N_occurrence]
graph_index_by_occurrence:  LongTensor[N_occurrence]
```

All mappings must be built from stable IDs and serialized with the training checkpoint or reproducible preprocessing artifacts. Do not assume database IDs are already contiguous or ordered consistently across runs.

Pseudocode:

```python
base = entity_id_embedding(entity_indices)           # [N_entity, d]
base = base + type_embedding(entity_type_indices)    # [N_entity, d]
base = base + project_frozen_text(text_features)     # [N_entity, d]
base = base + missing_text_embedding(text_present)   # [N_entity, d]
base = base + project_optional_attributes(attrs)     # [N_entity, d], if configured

entity_base = base                                    # one row per Entity
occurrence_base = entity_base[entity_index_by_occurrence]
delta = torch.zeros_like(occurrence_base)
z_occurrence = layer_norm(occurrence_base + delta)    # [N_occurrence, d]
```

The code must not create an `nn.Embedding(N_occurrence, d)` or a parameter keyed by graph-node occurrence. `graph_index_by_occurrence` is metadata for later stages; Stage 1 does not add a learned graph-specific identity vector.

## 6. Gradient and optimizer behavior

Trainable in Stage 1:

- Global entity identity table, if enabled.
- Entity-type embedding table.
- Text/attribute projection layers.
- Missing-feature embeddings/flags projection.
- LayerNorm parameters used by the canonical/occurrence initialization path.

Frozen in MVP 1:

- Precomputed biomedical text vectors or the upstream text encoder that produced them.

The final link-prediction loss must be able to backpropagate into all trainable Stage 1 parameters. Do not detach canonical or occurrence states before later stages. Frozen text features need no gradient; their projection does.

## 7. Edge cases and required behavior

| Case | Required behavior |
|---|---|
| Entity appears in ROOT and several internal graphs | One canonical row, one occurrence row per GraphNode |
| Entity has no text vector | Use zero text vector and missing-feature path; keep entity/type features |
| Entity type is unknown | Use shared `UNK` type embedding |
| Entity has no occurrence in a sampled minibatch | Its canonical row may be omitted from active computation; do not fabricate an occurrence |
| Duplicate occurrence mapping for the same GraphNode ID | Preprocessing error |
| Two GraphNodes for same entity and same graph | Preprocessing error under the GoG unique-per-entity-per-graph invariant |
| New entity at inference | Use type/text/attribute projections; do not require a training occurrence parameter |
| Empty ROOT/internal graph | No occurrence rows; no special trainable graph-node identity is created |

## 8. Leakage and reproducibility

Stage 1 uses entity-level features, not held-out relation structure. Maintain separate train/evaluation edge masks for later stages; Stage 1 must not materialize a feature that encodes the target relation labels or held-out neighborhood.

Save with the checkpoint:

- Entity and type index mappings.
- Feature schema and normalization parameters.
- Frozen text-feature source/version and preprocessing metadata.
- Whether the canonical ID embedding table is enabled.
- Hidden dimension and initializer configuration.

## 9. Acceptance criteria

Stage 1 is implemented correctly when:

1. There is exactly one canonical trainable identity row per global entity (if the ID table is enabled).
2. The model has no trainable occurrence identity lookup table.
3. Every occurrence row is gathered from its canonical entity row and transformed only by shared modules.
4. Duplicate appearances of the same entity map to the same canonical row.
5. Missing text/type inputs follow deterministic configured fallback paths.
6. Shapes remain `[N_entity, 256]` and `[N_occurrence, 256]` for the default dimension.
7. Gradients flow through trainable initialization projections from downstream prediction loss.
8. The output is reproducible for the same checkpoint, mappings, and features.

## 10. Handoff to Stage 2

Stage 2 receives:

```text
z_occurrence: [N_occurrence, 256]
entity_index_by_occurrence: [N_occurrence]
graph_index_by_occurrence: [N_occurrence]
internal_local_edge_index: [2, E_local]  # local occurrence indices
internal_local_edge_type: [E_local]
```

It updates occurrence states from same-internal-graph relational neighborhoods. It does not change the identity mapping or create per-occurrence parameters.
