# Graph-of-Graphs (GoG) Embedding Pipeline — Implementation Overview

**Purpose:** This document gives a coding agent the end-to-end architecture and implementation boundaries for MVP 1 of the biomedical Graph-of-Graphs embedding model. It is the overview document. Each stage is specified in a separate, more detailed Markdown file before that stage is implemented.

**Status:** Architecture overview / implementation brief  
**Model scope:** One ROOT knowledge graph with optional one-level internal graphs  
**Initial hidden size:** 256  
**Initial propagation rounds:** 2  
**Primary task:** Typed link prediction over ROOT, internal, and explicit cross-graph relations

---

## 1. Objective

Build a relational graph encoder that uses the hierarchy in the data rather than flattening all occurrences into an ordinary graph. It must learn from:

1. Relations within each internal graph.
2. Explicit relations whose endpoints lie in different graph contexts.
3. The internal structure associated with graph-valued entities.
4. Ordinary relations between entities in the ROOT KG.
5. Repeated occurrences of the same global entity across graph contexts.

The model must produce useful embeddings for canonical entities, contextual entity occurrences, and internal graphs. It must retain a single global identity per entity while allowing each occurrence to represent its local context.

## 2. Architecture in one view

```text
Canonical entities + graph/occurrence/relation data
                         │
                         ▼
             Initialize canonical states
                         │
             ┌───────────┴───────────┐
             │  For each round (×2)  │
             │                       │
             ▼                       │
   1. Build contextual occurrence states
             │
             ▼
   2. Local CompGCN inside each internal graph
             │
             ▼
   3. Sparse cross-graph attention on explicit cross-context edges
             │
             ▼
   4. Attention-pool internal graph representations
             │
             ▼
   5. Fuse each graph representation with its associated graph-valued entity
             │
             ▼
   6. ROOT CompGCN on ROOT-to-ROOT relations only
             │
             ▼
   7. Synchronize occurrence messages into canonical entity states
             │
             ▼
   8. Recompute contextual adjustments and feed updated context downward
             └───────────────────────┘
                         │
                         ▼
       Final occurrence/entity states → link-prediction decoder
```

The stages are ordered because information learned inside an internal graph must be able to affect its associated entity before that entity sends messages over ordinary ROOT relations. Identity synchronization and downward feedback then make updated information available to occurrences in the next round.

## 3. Data model

The encoder consumes the existing GoG schema. Do not add a second graph hierarchy or synthesize relations to represent ownership.

### 3.1 Core objects

| Object | Meaning | Key fields / relationship |
|---|---|---|
| `Entity` | One global real-world identity | `entity_id`, features/type; one canonical state per entity |
| `Graph` | ROOT or an internal graph | `graph_id`; internal graph has `owner_entity_id`; ROOT has no owner |
| `GraphNode` | One entity occurrence in one graph | `node_id`, `entity_id`, `graph_id` |
| `Relation` | One directed KG fact between exact occurrences | `source_node_id`, `relation_type_id`, `target_node_id` |

An entity can have a ROOT occurrence, occurrences in multiple internal graphs, or both. Each occurrence points to the same `entity_id`; it does not get its own trainable identity vector.

### 3.2 Ownership is not a fact

An internal graph's `owner_entity_id` records which entity the graph is associated with. It is metadata for graph/entity fusion. It must not be turned into a KG relation or inserted into an edge list.

For MVP 1, expansion depth is one: ROOT plus optional internal graphs. Do not recursively expand internal nodes into further graphs.

### 3.3 Relations always use exact occurrence endpoints

Every fact connects two specific `GraphNode` occurrences. Relation scope is derived from the endpoint `graph_id`s:

- ROOT → ROOT: handled by Stage 6.
- Same internal graph → same internal graph: handled by Stage 2.
- Different graph contexts: handled by Stage 3, including ROOT ↔ internal and internal ↔ internal.

Do not copy a generic ROOT fact into every internal graph containing one of its entities. Do not infer or create cross-graph facts. A relation between two contexts exists only when the source and target `GraphNode` endpoints explicitly identify those contexts.

## 4. State and embedding contract

### 4.1 Canonical entity state

Each global entity (v) has one canonical state (b_vin\mathbb{R}^{d}), with (d=256) initially. Initialize it from the entity's available typed/text features and learned feature projections. If frozen biomedical text features are used, document their source, preprocessing, and dimensional projection in the implementation configuration.

There is no independent trainable embedding table indexed by occurrence. In particular, do not create a free parameter vector for every `(entity_id, graph_id)` pair.

### 4.2 Contextual occurrence state

For entity (v)'s occurrence in graph (G), compute a contextual adjustment Δ(v,G) from the canonical state, graph context, and messages accumulated by the pipeline. The occurrence state is:

\[
z_{v,G}=\operatorname{LayerNorm}(b_v+\Delta(v,G)).
\]

The adjustment is computed by shared model modules and current graph data; it is not an independent trainable identity. The same entity can therefore have different contextual occurrence states while retaining one shared canonical identity.

### 4.3 Graph state

Each internal graph has a 256D graph representation computed from its updated member occurrence states using shared attention pooling and a shared update module. ROOT is processed by its own ROOT path; it is not pooled as if it were an internal graph.

### 4.4 Parameter sharing

Use shared encoder modules across entities, occurrences, and internal graphs. Relation transformation parameters may differ by processing path (local internal vs ROOT), but relation IDs retain a consistent vocabulary. Reuse the model's round parameters across the two initial rounds unless later stage specifications establish a justified exception.

## 5. End-to-end pipeline

The following is the locked stage order and responsibility. The detailed equations, tensor layouts, edge cases, and module-level pseudocode will be documented in the corresponding stage files.

### Stage 0 — Input preparation and validation

Prepare canonical entity IDs, graph IDs, occurrence IDs, owner mappings, relation types, and feature tensors. Build graph-local index maps and classify relation instances by the graph IDs of their exact endpoint occurrences. Validate that every relation endpoint resolves to a GraphNode and every internal graph has at most one owner for MVP 1.

Split data before message-passing adjacency is constructed. Validation/test target edges must not be present in any encoder input channel, including reverse computational channels.

### Stage 1 — Canonical initialization and contextual occurrence states

Initialize one canonical state per entity. Build occurrence states from the canonical state plus computed context adjustments. All occurrences of one entity begin from the same canonical identity but can acquire different graph-specific context through later stages.

### Stage 2 — Local relational propagation inside internal graphs

Apply a relation-aware CompGCN-style encoder to each internal graph's within-graph edges. Use basis-decomposed relation/direction transforms to control parameter growth. This stage handles only edges whose endpoint occurrences share the same internal graph context.

This stage updates internal occurrence messages/states; it does not pool the graph and does not process ROOT-to-ROOT or cross-context edges.

### Stage 3 — Sparse cross-graph relation attention

Process only explicit relation instances whose endpoints belong to different graph contexts. Use sparse, relation- and endpoint-conditioned multi-head attention with a learned gate. Messages update the exact source/target occurrence contexts participating in those relations.

Do not run dense attention over all nodes in all graphs. Do not treat graph ownership as a relation. Do not copy edges across contexts. Any reverse-direction message channel is a computational parameterization of the recorded directed fact, not a new biomedical fact; implement it once and avoid duplicate edge materialization.

### Stage 4 — Internal graph readout

For every internal graph, use shared attention pooling over its current member occurrence states, then a shared gated residual update to form one 256D graph representation. Include the previous graph context in the pooling/update as specified in the stage document. A one-node graph pools that node; an empty graph preserves its prior context or is skipped according to the stage's explicit contract.

The ROOT graph does not pass through this internal graph pooling stage.

### Stage 5 — Graph/entity fusion

Exchange information between each internal graph representation and its associated graph-valued entity using shared gated fusion. The upward path lets the graph's structural representation influence the associated entity state. The reverse path provides entity/canonical context to the graph and its members for the next propagation round.

Use the ownership mapping only to pair the internal graph and associated entity occurrence/state. Do not create a separate owner identity, add a synthetic fact, or duplicate the same graph summary across unrelated occurrences. The associated graph-valued entity must remain the same global entity identity wherever it occurs.

### Stage 6 — ROOT relational propagation

Apply a ROOT-specific CompGCN-style layer to ROOT-to-ROOT relation instances only. Use the current ROOT occurrence states, including the graph/entity fusion result for graph-valued entities. Ordinary atomic and graph-valued ROOT entities participate together.

ROOT ↔ internal relations are not processed here; Stage 3 handles those exact cross-context edges. Do not process the same ROOT edge twice or insert synthetic ownership edges. Use synchronous updates from one input snapshot per propagation round.

### Stage 7 — Global identity synchronization

Aggregate occurrence-level messages/states for each canonical entity across its observed graph occurrences with a shared, learned aggregation/update module. Update the single canonical state once per entity. Preserve context-specific occurrence information in the computed contextual adjustments rather than storing independent occurrence identities.

Entities with one occurrence are valid inputs. Define deterministic behavior for entities with no active occurrence in a sampled batch (retain their previous canonical state or omit them from the update).

### Stage 8 — Contextual adjustment and downward feedback

Use the updated canonical states and current graph contexts to recompute Δ(v,G) and contextual occurrence states. Feed updated entity/root information back into internal graph occurrences through shared gated modules. This is how information from the ROOT KG can affect internal contexts in the next round.

Do not simply overwrite all contextual states with the canonical vector; retain a computed, graph-specific adjustment. Do not use a separate learned identity vector per occurrence.

### Repeat and decode

Run the stages through two propagation rounds initially, reusing the configured shared parameters. The second round consumes the canonical/context/graph states produced by the first. The final encoder outputs feed the link-prediction decoder for the requested relation and endpoint contexts.

## 6. Link prediction and training boundary

The primary objective is typed link prediction. The model should be able to score relation triples at all observed scopes:

- ROOT endpoint pairs.
- Within-internal-graph endpoint pairs.
- Explicit cross-graph endpoint pairs.

Use the directional ComplEx-style decoder selected for this project, with relation scoring conditioned on the correct endpoint occurrences/contexts. The decoder and negative sampling must respect entity types and graph/context validity. Keep the same global entity identity for candidate identity while selecting the appropriate occurrence/context when a fact is context-specific.

Train end-to-end so the prediction loss can update the decoder, canonical/entity feature encoder, local and cross-graph modules, graph readout/fusion, ROOT encoder, identity synchronization, and feedback modules. Do not detach intermediate graph or occurrence states unless a later stage specification explicitly requires it.

### Leakage controls

- Remove each validation/test target fact from all message-passing inputs.
- Do not leave its generated reverse channel in the graph.
- Do not make held-out facts reachable by copying them into another graph context.
- Fit feature preprocessing, vocabulary, and other learned statistics on the permitted training split only.
- Keep train/validation/test graph construction and candidate filtering explicit in code and configuration.

## 7. Efficiency and scaling constraints

The architecture is intentionally selective:

- Store one trainable canonical vector per entity, not per occurrence.
- Compute contextual occurrence states from shared modules and active graph batches.
- Run local propagation on per-graph sparse edges.
- Run cross-graph attention only on explicit cross-context edges, not all node pairs.
- Use basis decomposition for relation-specific transforms.
- Use shared graph pooling/fusion/synchronization modules.
- Avoid materializing dense all-pairs attention or a separate (256\times256) matrix per relation instance.
- Track the number of entities, occurrences, graphs, local edges, cross-context edges, and peak active batch sizes during preprocessing and training.

## 8. Required output contract

At minimum, the encoder should expose:

1. **Canonical entity states** — one 256D state per global entity.
2. **Contextual occurrence states** — computed 256D states for `(entity_id, graph_id)` GraphNode occurrences.
3. **Internal graph states** — one 256D state per internal graph.
4. **Link scores** — typed scores for candidate relation instances using the correct graph-node endpoints.

Outputs must be traceable back to entity, graph, and occurrence IDs. Occurrence embeddings are model outputs, not additional independently stored identities.

## 9. Implementation workflow for the coding agent

1. Inspect the repository, current data schema, model code, training loop, and configuration before changing files.
2. Map the existing implementation to the objects and stage boundaries in this overview. Preserve compatible code and report mismatches clearly.
3. Implement only against the matching detailed stage specification once it is supplied. Do not guess unresolved tensor semantics or replace a stage with a flat-graph shortcut.
4. Keep each stage modular, with explicit input/output contracts and sparse index structures.
5. Make dimensions, round count, number of bases/heads, dropout, feature source, and sampling settings configurable; use the values in this overview as defaults only where fixed.
6. Document any unavoidable deviation from the architecture and its effect on information flow, identity sharing, or leakage.
7. Do not change the GoG data model, add synthetic relation types/edges, or create per-occurrence trainable identity parameters without an explicit architecture revision.

## 10. Non-negotiable invariants

- One canonical global identity per real entity.
- Occurrence states are computed as canonical identity plus graph-specific context; no independent trainable occurrence identity table.
- Relations connect exact GraphNode occurrences and remain directed facts.
- Ownership is association metadata, never a KG fact.
- One-level internal graph expansion for MVP 1.
- Local internal edges, explicit cross-context edges, and ROOT-to-ROOT edges are handled by their designated stages exactly once.
- Cross-graph attention is sparse and restricted to explicit cross-context relations.
- Information flows from internal structure to associated entities and from updated higher-level/canonical context back to internal occurrences over rounds.
- Held-out facts are absent from every message-passing route.

## 11. Follow-up stage documents

We will produce separate implementation documents, in order:

1. `00_data_contract_and_preprocessing.md`
2. `01_canonical_initialization_and_occurrence_context.md`
3. `02_internal_local_compgcn.md`
4. `03_sparse_cross_graph_attention.md`
5. `04_internal_graph_readout.md`
6. `05_graph_entity_fusion.md`
7. `06_root_compgcn.md`
8. `07_global_identity_synchronization.md`
9. `08_contextual_adjustment_and_downward_feedback.md`
10. `09_decoder_training_and_leakage_safe_evaluation.md`

Stage 0 defines a target schema and dataset-adapter boundary, not a commitment to a particular source KG. A flat source KG may be used as ROOT-only data unless external metadata supplies valid internal graph contexts and ownership.

Each stage document should define: purpose, exact tensor/data inputs and outputs, equations and dimensions, parameter sharing, update order, sparse implementation approach, edge cases, gradient flow, leakage constraints, pseudocode, and acceptance criteria. If a detail in a stage document conflicts with this overview's invariants, call out the conflict and resolve it before implementation.

---

## Short version for the coding agent

Implement an identity-aware hierarchical relational GNN over the existing GoG model. Keep one canonical embedding per global entity and compute contextual occurrence states from shared canonical-plus-context modules. Per round, process internal local edges, explicit sparse cross-context edges, pool internal graph states, exchange graph/entity information, process ROOT-to-ROOT facts, synchronize repeated occurrences into canonical identities, and feed updated context back down. Repeat for two rounds and train with leakage-safe typed link prediction. Ownership is not a relation; relation endpoints are exact GraphNode occurrences; never flatten away context or create synthetic facts.
