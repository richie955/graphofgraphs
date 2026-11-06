# Stage 0 — Data Contract and Preprocessing

**Purpose:** Define the dataset-independent input schema and deterministic preprocessing contract for the GoG encoder. Dataset-specific importers map source data into this contract; the model never depends on a particular source KG format.

**Status:** Canonical in-memory schema. The loader accepts both the synthetic fixture and the simplified PrimeKG GoG JSON in `docs/primekg_gog_30000.json`.

---

## 1. Boundary and principles

Stage 0 converts source records into canonical entities, graph contexts, entity occurrences, directed facts, feature tensors, and split-specific sparse edge lists. It does not infer graph ownership, copy facts into contexts, generate semantic inverse facts, or create model parameters.

The source adapter must provide the GoG hierarchy or an explicit documented mapping rule. A flat KG alone does not identify internal graphs, graph owners, or context-specific occurrences. If it cannot supply those semantics, use it as a ROOT-only dataset for an initial encoder/decoder baseline, or choose a source with context metadata; do not fabricate a hierarchy merely to satisfy the schema.

The PrimeKG GoG JSON uses `entities[].type`, `graphs[].type`, `graphs[].parent_entity_id`, and a top-level `relations` table. Stage 0 accepts these names directly: entity `type` becomes the entity type, graph `type` becomes its kind, `parent_entity_id` becomes the owner, and relations become the routed facts. A relation's model ID combines its broad `type` and, when present, its `display_relation` (for example, `drug_protein::target`) so distinct labels are retained. The input file is not rewritten.

## 2. Canonical in-memory schema

| Object | Required fields | Constraints |
|---|---|---|
| Entity | `entity_id`, optional `type_id`, features | One row per global identity; IDs stable within a dataset version |
| Graph | `graph_id`, `kind`, optional `owner_entity_id` | Exactly one `ROOT`; other graphs are internal; no nested graphs in MVP 1 |
| GraphNode | `node_id`, `entity_id`, `graph_id` | One exact entity occurrence in one context; node ID unique |
| Relation | `fact_id`, `source_node_id`, `relation_type_id`, `target_node_id` | Directed fact with exact occurrence endpoints |

Relations may include an optional source provenance key, confidence, or time field, but such fields are not model inputs unless an explicit feature policy is configured. Ownership is graph metadata, never a relation.

For MVP 1, each internal graph must resolve to exactly one owner entity and exactly one ROOT occurrence for that owner; each owner has at most one internal graph. Treat violations as data-contract errors until an architecture extension is specified. ROOT itself has no owner. Empty internal graphs may be represented, but Stage 5 skips their fusion.

The GoG model contract allows at most one GraphNode for each `(entity_id, graph_id)` pair. Reject repeated pairs as invalid input; the same entity appearing once in ROOT and once in an internal graph is expected and is not a duplicate. This unique-per-graph invariant makes candidate occurrence lookup deterministic.

## 3. Source adapter contract

A dataset adapter must document:

- Source name, version, license/access conditions, and source-to-target field mappings.
- How global entity identity is established across source tables and contexts.
- How ROOT/internal graph membership, ownership, and occurrence IDs are obtained.
- How relation direction, relation vocabulary, duplicates, qualifiers, and explicit inverse predicates are handled.
- Which entity features are available at inference and how missing values are represented.
- Whether the data is static or time/version scoped.

If ownership, occurrence context, or cross-context relation endpoints cannot be mapped unambiguously, fail preprocessing with a useful error. Do not join relation endpoints by entity ID alone when multiple context occurrences could match.

## 4. Indexing and tensor outputs

Build deterministic contiguous mappings from sorted stable IDs (or a documented stable ordering):

```text
entity_id_to_index: Entity ID -> [0, N_entity)
graph_id_to_index:  Graph ID  -> [0, N_graph)
node_id_to_index:   GraphNode ID -> [0, N_occurrence)
relation_id_to_index: relation ID -> [0, N_relation_type)
entity_index_by_occurrence: LongTensor[N_occurrence]
graph_index_by_occurrence:  LongTensor[N_occurrence]
root_occurrence_by_entity: LongTensor[N_entity], plus has_root_occurrence mask
owner_entity_by_graph: LongTensor[N_internal_graph], plus has_owner mask
owner_root_by_graph: LongTensor[N_internal_graph], plus has_owner_root mask
owned_graph_by_entity: LongTensor[N_entity], plus has_owned_graph mask
```

Use explicit masks for absent mappings; never use index zero as a missing sentinel. Preserve reverse mappings and raw IDs for traceability. Serialize mappings, vocabularies, preprocessing configuration, and a dataset/version fingerprint with checkpoints.

After split assignment, emit separate fact records and sparse index arrays for `internal_local`, `cross_context`, and `root` facts. Every edge row retains `fact_id`, relation ID, and exact source/target occurrence indices. Classify scope only from endpoint graph IDs:

- same non-ROOT graph → Stage 2;
- different graph IDs → Stage 3;
- both ROOT → Stage 6.

No fact may occur in more than one scope list. Ownership never enters any edge list.

## 5. Splits and leakage-safe adjacency

Assign each semantic fact to train, validation, or test **before** building any message-passing adjacency. Keep the mapping from semantic fact to all its computational channels so a held-out fact and its generated channels are excluded together. If inverse predicates are supplied as separate facts, preserve them as source facts and define whether they are grouped into one split unit; do not both preserve and synthesize the same inverse evidence.

The PrimeKG GoG file has no split labels, so the loader assigns deterministic approximate 80/10/10 splits (seed 42 by default; configurable through `load_gog` or `train_stage9.py --split-seed`). It groups all facts between the same unordered pair of canonical entities before splitting, across ROOT and internal contexts and across relation labels. This keeps a held-out link from appearing as a training link through another graph occurrence. Groups with fewer than three units remain in training. The report records whether splits were generated and the seed. This is a plumbing split, not a benchmark-prescribed evaluation protocol. A real benchmark may require a different split policy; record and use that policy instead. Ensure each evaluated target has valid endpoint occurrences and define handling for entities/types absent from training.

Relation endpoint type rules are optional. If a dataset supplies them, Stage 0 validates them and Stage 9 uses them to restrict negative and ranking candidates. If no rules are supplied, candidates are limited by the required graph context only; this broader candidate policy is recorded in the preprocessing report.

Construct encoder adjacency from training facts only. Validation/test facts are labels/candidates for evaluation, never encoder edges. Build forward and reverse **computational channels** inside the owning model stage from allowed facts only. Do not add those channels to the semantic fact table.

Fit feature normalization and any learned vocabularies/statistics on training-available data only. Frozen pretrained text vectors may be precomputed, but document model/version and ensure text construction does not encode target relation neighborhoods or labels.

## 6. Validation checks

Fail preprocessing with counts and sample offending IDs when any of these checks fails:

1. IDs are unique within their object tables; every foreign key resolves.
2. Exactly one ROOT graph exists; graph depth and kind obey the MVP contract.
3. Every relation endpoint resolves to exactly one GraphNode, and its stored graph scope agrees with endpoint graph IDs.
4. Owner constraints in Section 2 hold, including an unambiguous ROOT owner occurrence.
5. No fact is routed to multiple stage edge lists; no ownership edge exists.
6. Duplicate facts and duplicate occurrence rows follow configured policies.
7. Split sets are disjoint by semantic fact/split unit; held-out facts are absent from all encoder channels, including reverse channels.
8. Relation/type IDs, feature dimensions, finite values, and missing-value masks are valid.
9. Every evaluation target can resolve its relation and endpoint context; negative candidates can be drawn under the declared candidate rules.

Produce a preprocessing report with counts for entities, types, graphs, internal graphs, occurrences, owners, relations by scope and split, duplicates, feature coverage, and validation failures.

## 7. Batching contract

Initial MVP should use full-graph training adjacency when it fits memory. If sampling is required, the sampler must return a closed encoder subgraph with exact occurrence mappings, split-safe edges, graph memberships, and owner/root mappings. Stage 7 must aggregate all active occurrences for a canonical entity before one GRU update; batches that split those occurrences must accumulate their synchronization inputs before updating, or use an entity-complete batch policy.

Record whether each pass uses the full training graph or sampled neighborhoods. Evaluation must use a deterministic, documented neighborhood policy and the same split-safe adjacency construction.

## 8. Synthetic fixture before source selection

Use `synthetic_gog_fixture.json` and its README as the reference fixture. `gog_preprocessing.py` implements the dependency-free loader, validation, indexing, training-fact routing, report, and CLI. Run `python -m unittest -v` for contract checks or `python gog_preprocessing.py docs/synthetic_gog_fixture.json` for a preprocessing report. It is a software fixture, not a claim about biomedical data. Empty/singleton graph behavior remains covered by small stage-level test cases.

## 9. Acceptance criteria and handoff

Stage 0 is ready when the adapter emits validated tables, stable mappings, scope-partitioned sparse facts, split masks, feature tensors/masks, and a reproducible report. All downstream stages consume these canonical tensors and IDs; they must not parse source-specific files or infer missing hierarchy semantics.

Stage 1 receives entity/type/features, occurrence-to-entity and occurrence-to-graph indices, and the serialized mappings. Stages 2, 3, and 6 receive only their respective training fact arrays. Stages 5 and 8 receive the validated owner/root maps.
