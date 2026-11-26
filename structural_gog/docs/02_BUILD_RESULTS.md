# Biomedical Graph-of-Graphs — Results

Built 2026-11-26 from PrimeKG (8,100,498 edges, 129,312 nodes), PubChem, Reactome,
UniChem and UniProt. All validation checks pass. No model has been trained yet.

---

## 1. What was built

### Level 1 — root graph (PrimeKG subset)

| | |
|---|---|
| nodes | **2,534** |
| edges | **40,944** |

gene/protein 1,118 · disease 592 · effect/phenotype 417 · drug 247 · pathway 160

Selection rule: seed on drugs with a usable molecular structure and Reactome human
pathways of 10–200 nodes, expand 1 hop, keep the largest connected component.

### Level 2a — drug internal graphs

| | |
|---|---|
| graphs | **246** |
| nodes | 2,882 |
| edges | 2,390 |
| validation | **246/246 molecules reconstructed exactly** |

Each molecule is decomposed into ring systems, functional groups and loose atoms.
32 distinct fragment labels; the commonest are `aromatic_ring` (232),
`ring` (111), `amide` (72), `carboxylic_acid` (58), `phosphate` (23),
`sulfonamide` (16).

### Level 2b — pathway internal graphs

| | |
|---|---|
| graphs | **124** |
| nodes | **6,747** |
| edges | **10,185** |
| reactions | 2,077 |

Nodes: molecular_state 2,615 · reaction_event 2,077 · complex_state 1,473 ·
set_state 582
Edges: input_to 3,856 · produces 3,293 · precedes 1,661 · catalyzed_by 921 ·
activated_by 270 · inhibited_by 184

Mean 54.4 nodes per pathway. Of 160 candidates: 124 accepted, 20 too large,
16 too small.

### Cross-level links

370 owner links (root node → its internal graph) and **4,693 cross-context edges**:

| edge type | count | evidence |
|---|---|---|
| `same_entity_as` | 4,267 | shared UniProt accession — grounds pathway entities in root proteins |
| `targets_reaction` | 329 | the drug's PrimeKG target protein participates in the reaction |
| `participates_in_reaction` | 75 | the drug itself is a curated reaction participant (ChEBI) |
| `same_chemical_as` | 22 | whole-molecule reference ↔ Reactome molecular state |

**28 edges were dropped by the leakage filter** for re-encoding a held-out
`(drug, target protein)` fact.

### Structural gain

> **9,629 internal nodes and 12,575 internal edges on top of 2,534 root nodes — 3.8×.**

None of this is representable in PrimeKG: it has no atoms, no bonds, no functional
groups, no reactions, no compartments, and no molecular states.

---

## 2. Evaluation splits

### Primary splits — 8:1:1 over the hybrid universe

`balanced_{train,validation,test}_facts.json`, stratified per relation:

| split | facts | share | touching an internal graph |
|---|---|---|---|
| train | **96,034** | 80.21% | 10,001 (10.4%) |
| validation | **11,823** | 9.88% | 1,238 (10.5%) |
| test | **11,869** | 9.91% | 1,240 (10.4%) |
| total | 119,726 | | |

Per-relation test shares land between 8.6% and 10.5%. Held-out counts for the
relations that matter: `pathway_protein` 149, `drug_protein` 164,
`contraindication` 269, `indication` 56. Four relations with 8 or fewer edges stay
wholly in train.

`balanced_{validation,test}_gog_slice.json` hold the 1,238 and 1,240 held-out facts
whose head or tail owns an internal graph. **Report metrics on these slices
alongside the full split** — the full split measures the model, the slice isolates
what the graph-of-graphs contributes.

An alternative pair of splits also exists (`{train,validation,test}_facts.json`,
16,457 / 2,003 / 2,009) restricted entirely to the GoG subset, where 34% of held-out
facts touch an internal graph rather than 10%. Smaller and more concentrated; useful
as a second configuration.

### Versus the original flat splits

| | original | this work |
|---|---|---|
| train / validation / test | 160k / 20k / 20k | 96,034 / 11,823 / 11,869 |
| ratio | 80 / 10 / 10 | 80.2 / 9.9 / 9.9 |
| each edge scored twice | **yes** | no |
| source identifiers | **absent** | present |
| `drug_drug` share of test | **80.0%** | 62.3% (9.0% in subset splits) |
| `pathway_protein` test edges | **2** | **149** |
| `indication` test edges | 40 (20 unique) | 56 |
| leakage verified | asserted | **15 checks, from disk** |

The original splits handled inverse-relation leakage correctly, which is the hard
part. Their problems were double-counted edges, missing source identifiers, and a
relation mix dominated by `drug_drug` — with `pathway_protein` reduced to a single
test edge, making Phase 3 unmeasurable.

### How leakage is ruled out

Four mechanisms, the third specific to a graph-of-graphs:

1. **Direction pairs cannot separate.** PrimeKG stores every edge twice, so a naive
   random split puts `(a,r,b)` in train and `(b,r,a)` in test. Edges are collapsed to
   one record per undirected key before assignment, which also stops metrics from
   counting each edge twice.
2. **Assignment happens on the collapsed set**, so an edge is assigned exactly once.
3. **Cross-context edges are filtered against the held-out sets.** A
   `targets_reaction` edge is derived *from* a PrimeKG `(drug, protein)` fact; if that
   fact is held out, the edge is a two-hop copy of the answer via
   drug → reaction → protein. **16 edges were dropped.** The filter re-runs whenever
   the held-out sets change.
4. **Verified by reading the written files**, not from build-time state:

```
pass  train/validation   exact · reversed · undirected    0  0  0
pass  train/test         exact · reversed · undirected    0  0  0
pass  validation/test    exact · reversed · undirected    0  0  0
pass  internal duplicate edges, all three splits          0
pass  cold-start nodes in validation / test               0
pass  341 targets_reaction edges vs 47,382 held-out pairs 0
```

**Not checked:** leakage paths of three or more hops. A route through the pathway
level that reconstitutes a held-out fact in several steps would not be caught. Two-hop
re-encoding is filtered and independently verified; deeper paths are not.

---

## 3. Ready for the model

`data/out/tensors/` holds `.npz` arrays and a `manifest.json`:

- drug graphs: 246, feature dim 41 (34-dim atom features, 32-label fragment one-hot)
- pathway graphs: 124, feature dim 37 (schema class, multi-hot compartment, identifier flags)
- root edges as COO index per relation, per split
- `pip install torch torch-geometric` also produces `gog_pyg.pt` with a `HeteroData`
  root plus per-owner `Data` objects

---

## 4. Honest limitations

1. **No model has been run.** Whether this structure improves link prediction is
   untested. Morgan fingerprints on drug nodes are the baseline to beat; if the
   fragment graphs cannot beat them, the drug level is not earning its complexity.
2. **Cross-context edges are 91% grounding edges.** `same_entity_as` dominates at
   4,267; genuine drug-level participation is only 97 edges. Reactome simply does
   not curate many drugs as reaction participants.
3. **Guide to Pharmacology matching produced nothing.** 40 drugs carry a GtP id from
   UniChem, but none coincided with a GtP-typed entity in the 124 built pathways, and
   name matching added none either. Every direct edge came from ChEBI. More pathways
   would likely change this.
4. **Fragment collapsing is lossy.** One 22-heavy-atom drug reduces to 6 nodes, 16
   atoms inside a single ring node. Full atom detail is preserved in the fragment
   attributes, so a less aggressive variant needs no rebuild — but as exported, a GNN
   sees coarse structure.
5. **Small evaluation.** 2,009 test facts, of which 22 are `indication`. Enough for
   aggregate metrics, too few for per-relation claims about rare relations.
6. **Coverage is a sample, not the whole of PrimeKG.** Built from 300 sampled drugs
   and 80 sampled pathways. Scaling to all 7,957 drugs costs ~4 h and all 2,516
   pathways ~24 h, both by changing one number in `config.yaml`.
7. **11 internal graphs were discarded** because their owner did not survive the
   largest-connected-component step.

---

## 5. Reproducing

```bash
python scripts/01_coverage_check.py        # ~9 min   identifier resolution
python scripts/02_build_root_graph.py      # ~10 min  subset + root graph
python scripts/03_make_splits.py           # seconds  leakage-free splits
python scripts/04_build_drug_graphs.py     # seconds  246 molecular graphs
python scripts/05_build_pathway_graphs.py  # ~56 min  124 pathway graphs
python scripts/06_build_cross_edges.py     # ~2 min   cross-context edges
python scripts/07_integrate.py             # seconds  biomedical_gog.json
python scripts/08_validate.py              # seconds  validation report
python scripts/09_export_tensors.py        # seconds  tensors
python scripts/10_make_hybrid_splits.py    # ~2 min   absorb flat-file facts
python scripts/11_make_balanced_splits.py  # seconds  8:1:1, verified leak-free
python scripts/06_build_cross_edges.py --splits balanced   # re-filter
python scripts/07_integrate.py && python scripts/08_validate.py
python scripts/09_export_tensors.py --splits balanced
```

Every HTTP response is cached under `data/cache/`, so a re-run takes minutes and the
build is reproducible offline.
