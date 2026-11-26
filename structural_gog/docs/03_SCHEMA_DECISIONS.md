# Model-input schema — for sign-off before building

Target: the four flat tables in `graphofgraphs/docs/00_data_contract_and_preprocessing.md`,
consumed by `load_gog()` and `python -m training.train <file> --model compare`.

Two formats, two jobs. `biomedical_gog.json` stays as the construction artifact — nested,
with provenance, atom indices, internal bonds and evidence dicts, so the build is
auditable. The model file below is flat and carries none of that.

---

## 1. The four tables

```json
{
  "entities":    [ {"entity_id", "name", "type", "source", "source_id"} ],
  "graphs":      [ {"graph_id", "type", "parent_entity_id"} ],
  "graph_nodes": [ {"node_id", "entity_id", "graph_id"} ],
  "relations":   [ {"relation_id", "source_node_id", "type",
                    "target_node_id", "display_relation", "split"} ],
  "metadata":    { ... }
}
```

`node_id` is `"<entity_id>@<graph_id>"`. No entity id contains `@`, so the key is
unambiguous. Scope (`root` / `internal` / `cross_graph`) is **not** written — their
loader derives it from endpoint graph ids, and writing it would be a second source
of truth.

Ownership appears only as `graphs[].parent_entity_id`, never as a relation. Their
contract requires this.

---

## 2. Entity types

| `type` | count | origin |
|---|---|---|
| gene/protein | 3,725 | PrimeKG root |
| drug | 3,366 | PrimeKG root |
| disease | 1,078 | PrimeKG root |
| effect/phenotype | 529 | PrimeKG root |
| pathway | 285 | PrimeKG root |
| biological_process | 249 | PrimeKG root |
| anatomy | 147 | PrimeKG root |
| molecular_function | 106 | PrimeKG root |
| cellular_component | 97 | PrimeKG root |
| exposure | 11 | PrimeKG root |
| **atom** | 1,917 | RDKit, per drug |
| **molecular_fragment** | 719 | RDKit, per drug |
| **reaction_event** | 2,077 | Reactome |
| **molecular_state** | ~2,615 occ. | Reactome |
| **complex_state** | ~1,473 occ. | Reactome |
| **set_state** | ~582 occ. | Reactome |

The six bold types are the structure PrimeKG cannot express. That is the entire
contribution, and it is visible in `entities[].type` alone.

Entity ids keep their readable form so any row traces back:

```
root:drug:DB00945            root entity
drug:DB00945::atom:7         atom (unique to that drug)
drug:DB00945::frag:0         fragment
reactome:R-HSA-113595        pathway entity, keyed by Reactome stId
```

---

## 3. Graphs

```
ROOT                      type "root",     no owner
G_drug_DB00945            type "internal", parent_entity_id root:drug:DB00945
G_pathway_R-HSA-111457    type "internal", parent_entity_id root:pathway:R-HSA-111457
```

1 ROOT + 370 internal (246 drug, 124 pathway). Every owner has exactly one internal
graph and exactly one ROOT occurrence, as the MVP contract requires.

---

## 4. Relations

| `type` | count | split | scope derived as |
|---|---|---|---|
| 28 PrimeKG relations | 119,726 | **train / validation / test** | root |
| `chemical_bond` | 879 | train | internal |
| `chemical_attachment` | 1,511 | train | internal |
| `input_to` | 3,856 | train | internal |
| `produces` | 3,293 | train | internal |
| `precedes` | 1,661 | train | internal |
| `catalyzed_by` | 921 | train | internal |
| `activated_by` | 270 | train | internal |
| `inhibited_by` | 184 | train | internal |
| `participates_in_reaction` | 75 | train | cross_graph |
| `targets_reaction` | 341 | train | cross_graph |
| `same_entity_as` | 4,267 | train | cross_graph |
| `same_chemical_as` | 22 | train | cross_graph |

**Only PrimeKG root facts carry held-out labels.** Structural and cross-context edges
are all `train`, i.e. encoder input only, never prediction targets. This follows §4 of
the handoff: derived or structural channels must be excluded from validation/test
targets. Internal link prediction is a separate task and is out of scope tonight.

`type` and `display_relation` are both passed through unchanged — their loader
composes the model relation id itself (e.g. `drug_protein::target`).

---

## 5. Decisions needing your sign-off

### D1 — Pathway entities keyed by Reactome stId ✅ recommended

6,747 occurrences collapse to **4,353 entities**. Ubiquitin becomes one entity with
occurrences in 20 pathway graphs instead of 20 unrelated entities.

Why: this is precisely what Stage 7 (global identity synchronisation) exists for — one
canonical state, many contextual states. Keying per occurrence would leave Stage 7
with nothing to synchronise. Verified safe: no pathway graph contains the same stId
twice, so the one-GraphNode-per-`(entity, graph)` invariant holds.

### D2 — ❌ REJECTED BY THE DATA — keep entities separate

**Measured:** 94 of 124 pathway graphs contain two or more entities that map to the
same root protein. For example `R-HSA-75893` has four distinct states sharing UniProt
P01375: `TNF(1-233) trimer [plasma membrane]`, `TNF-alpha trimer [extracellular
region]`, `TNF-alpha:TNFR1 [plasma membrane]`, `TNF-alpha inhibitors:TNF(1-233)
trimer`.

Merging would either violate the one-GraphNode-per-`(entity, graph)` invariant, which
their loader rejects, or collapse those states into one node — destroying the
compartment and complex structure that is the point of the pathway level. A partial
merge is impossible because `entity_id` is global: one stId cannot resolve to a root
protein in one pathway and stay separate in another.

**Decision: keep separate.** `same_entity_as` (4,267) and `same_chemical_as` (22) ship
as relations with `split: train`. Each carries a verified UniProt or ChEBI accession
match, so they are independently evidenced input channels — not the derived copies
handoff §4 objects to. `same_chemical_as` is retargeted from `drug::mol` to the root
drug entity, so D3 still holds. Stage 7 still has work via D1's stId keying.

<details><summary>Original recommendation, kept for the record</summary>

Where a pathway entity's UniProt accession resolves to a root protein, **use the same
`entity_id`** rather than two entities joined by a relation. Ubiquitin is then one
entity with occurrences in ROOT *and* in 20 pathway graphs.

Consequences:
- `same_entity_as` (4,267) and `same_chemical_as` (22) **disappear as relations** —
  identity moves into `graph_nodes`, which is where their schema puts it
- relation count drops to ~132,700
- `participates_in_reaction` and `targets_reaction` stay as relations: those are
  participation claims, not identity claims

Why: their own builder does this (handoff §1, lines 709-722: *"Each internal member
also receives an occurrence in ROOT. This is intentional shared entity identity"*).
Identity-as-a-relation would make the model learn something the schema can state
outright.

Partial by necessity: only 2,810 root proteins resolve to UniProt, and Reactome small
molecules often have no root counterpart. Unmerged entities stay separate — correct,
since no verified identity exists.

</details>

### D3 — Drop `molecule_ref` ✅ recommended

246 edge-free anchor nodes that exist only so `same_chemical_as` has a drug-side
endpoint. Under D2 that relation disappears, so these become 246 isolated entities
earning nothing. Dropped.

### D4 — The 11 conflicting entity pairs → train ✅ recommended

Their split unit is the unordered entity pair, and a pair spanning two splits is a hard
error. 11 pairs do (e.g. `contraindication` in train, `off-label use` in validation).
Move all their facts to train — removes them from evaluation, which is the safe
direction.

### D5 — Emit both split configurations ✅ recommended

| file | root entities | train / val / test | test touching an internal graph |
|---|---|---|---|
| `gog_model_balanced.json` | 9,593 | 96,034 / 11,823 / 11,869 | 10.4% |
| `gog_model_subset.json` | 2,534 | 16,457 / 2,003 / 2,009 | **34.1%** |

Run the subset file first: smaller, faster, and far more GoG signal. The balanced file
is the bigger benchmark where `drug_drug` is 62% of test.

---

## 6. Known limitation to state, not hide

Subset selection ran over **all** of `kg.csv` and splitting came afterwards, so graph
construction saw edges that later became test facts. Handoff §3.4 and §7 require
train-only construction inputs. This is selection leakage — mild, but real, and it
would fail their audit as written.

The fix is a reordering, not a rebuild: split source facts first, expand 1-hop over
train edges only. Not attempted tonight; stated as the next step.

---

## 7. Validation before anything is uploaded

The adapter output is checked by **their own** `load_gog()`, which enforces:

1. unique ids, every foreign key resolves
2. exactly one ROOT; owners have no owner; depth obeys the MVP contract
3. every relation endpoint resolves to exactly one GraphNode, scope agrees with endpoints
4. each internal graph has one owner with one unambiguous ROOT occurrence
5. no fact routed to two scope lists; no ownership edge present
6. splits disjoint by entity pair; held-out facts absent from every encoder channel
7. valid relation ids, finite features, valid masks

It is dependency-free, so this runs locally before Colab.
