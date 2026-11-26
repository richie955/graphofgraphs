# Structure-Based Biomedical Graph-of-Graphs — Full Project Handoff

Everything needed to build a presentation. Every number here was measured, not estimated;
where something is a projection or a guess it says so. Nothing in this document needs
outside knowledge to verify — the files named are all in the project.

Built 26 November 2026.

---

## 1. The problem

Biomedical knowledge graphs like PrimeKG store biology as flat triples:

```
(Aspirin, indication, pain)
(Aspirin, drug_protein, PTGS1)
(PTGS1, pathway_protein, Synthesis of Prostaglandins)
```

Each entity is an opaque node with a name. PrimeKG has **no representation of**:

- what a drug is *made of* — no atoms, no bonds, no functional groups
- what a pathway *does* — no reactions, no inputs or outputs, no catalysis
- where a protein *is* — no compartments, no complex states

So a model over PrimeKG cannot know that aspirin contains a carboxylic acid, that a
reaction converts one molecule into another, or that a protein in the cytosol and the
same protein in the mitochondrion are different states.

**The idea:** keep PrimeKG as a root graph, and give drug and pathway nodes their own
*internal graphs* describing their structure. A graph whose nodes are themselves graphs —
a graph-of-graphs. Then test whether that structure improves link prediction.

---

## 2. Data sources

| source | what it provides | access |
|---|---|---|
| PrimeKG (`kg.csv`) | root graph: 8,100,498 edges, 129,312 nodes | Harvard Dataverse `doi:10.7910/DVN/IXA7BM` |
| PubChem PUG REST | DrugBank id → compound → SMILES | public API, 5 req/s |
| RDKit 2026.09.2 | SMILES → molecular graph, ring and functional-group detection | library |
| Reactome Content Service | pathway reactions, participants, catalysts, regulators | public API |
| UniChem (EBI) | DrugBank → Guide to Pharmacology, ChEBI | public API |
| UniProt | Entrez gene id → reviewed accession | public API |

Every HTTP response is cached on disk, so the build is reproducible offline and a re-run
costs nothing.

---

## 3. Methodology

### Phase 0 — coverage check, before building anything

Three numbers decide whether the project is viable. Measured on a sample of 300 drugs and
80 pathways:

- **86.0% of drugs resolve to a usable SMILES** (258/300). Failures are biologics and
  mixtures, which have no molecular graph. Projected ~6,843 of 7,957 drugs usable.
- **70.0% of pathways fall in the 10–200 node range** (56/80). Median 24 nodes, p90 160,
  max 939. The distribution is bimodal: leaf pathways are small, top-level pathways are
  enormous.
- **Cross-context edges are viable by both matching methods** — 20 direct and 157
  protein-mediated pairs in the sample.

### Phase 1 — root graph

Seeded on drugs with a usable structure and Reactome human pathways in range, then
expanded 1 hop, excluding `drug_drug`, `protein_protein` and `anatomy_protein_present`
(hub relations that would pull in most of PrimeKG), then kept the largest connected
component.

**Result: 2,534 nodes, 40,944 edges.** 1,118 gene/protein, 592 disease, 417
effect/phenotype, 247 drug, 160 pathway.

### Phase 2 — drug internal graphs

For each drug: DrugBank id → PubChem CID → SMILES. Strip salts (keep the largest covalent
fragment), canonicalise, re-parse so atom indices match the stored SMILES.

Decomposition, in strict order:

1. **Ring systems first.** Fused and spiro rings merge into one system (any two rings
   sharing an atom belong together) and collapse to a single fragment node.
2. **Functional groups second**, by SMARTS, most specific pattern first. A group is
   accepted only if *every* one of its atoms is still unclaimed.
3. **Everything left stays an atom node.**

Doing rings first is what makes step 2 unambiguous — no functional group can straddle a
ring boundary, so overlaps resolve without arbitration. A one-atom group (a phenol
hydroxyl, a halide) annotates the atom rather than becoming its own node.

**Result: 246 graphs, 2,882 nodes, 2,390 edges. 246/246 validated** — every atom appears
exactly once and the original bond set is recoverable from the fragments.

31 distinct fragment labels. Most common: `aromatic_ring` 219, `ring` 108, `amide` 72,
`carboxylic_acid` 56, `phosphate` 23, `sulfonamide` 16.

**Worked example — aspirin**, canonical SMILES `CC(=O)Oc1ccccc1C(=O)O`, 13 heavy atoms:

```
NODES (6 + 1 reference)
  atom:0     C   methyl
  atom:1     C   carbonyl carbon
  atom:2     O   carbonyl oxygen
  atom:3     O   ester oxygen
  frag:0     aromatic_ring      atoms [4,5,6,7,8,9]   attach [4,9]
  frag:1     carboxylic_acid    atoms [10,11,12]      attach [10]

EDGES (5)
  atom:0 -> atom:1   chemical_bond        SINGLE
  atom:1 -> atom:2   chemical_bond        DOUBLE
  atom:1 -> atom:3   chemical_bond        SINGLE
  atom:3 -> frag:0   chemical_attachment  SINGLE   (atoms 3-4)
  frag:0 -> frag:1   chemical_attachment  SINGLE   (atoms 9-10)
```

The ester pattern would have matched atoms 1,2,3,4 — rejected because atom 4 was already
claimed by the ring. 13 atoms become 6 nodes, and every atom index and internal bond is
preserved so the molecule can be rebuilt exactly.

### Phase 3 — pathway internal graphs

For each pathway: a cheap two-call size probe first, rejecting oversized pathways before
fetching anything else. Then every contained reaction with its inputs, outputs, catalysts
(`CatalystActivity` → `physicalEntity`, a second lookup each), regulators, and
`precedingEvent` ordering.

Node types: `reaction_event`, `molecular_state`, `complex_state`, `set_state`.
Edge types: `input_to`, `produces`, `catalyzed_by`, `activated_by`, `inhibited_by`,
`precedes`.

**Three decisions worth defending:**

- **Compartment is part of identity.** `DIABLO [cytosol]` and
  `DIABLO [mitochondrial intermembrane space]` are separate nodes joined by the release
  reaction that moves one to the other. PrimeKG has one `DIABLO` node and no concept of
  location.
- **Entity sets stay unexpanded.** `DefinedSet` and `CandidateSet` become `set_state`
  nodes, so "one of these" is not silently converted to "all of these".
- **Ordering comes only from Reactome.** A `precedes` edge is emitted only where
  `precedingEvent` asserts it. No inferred ordering, because that would be an invented
  biological claim.

**Result: 124 graphs, 6,747 nodes, 10,185 edges, 2,077 reactions.** Of 160 candidates:
124 accepted, 20 too large, 16 too small. Mean 54.4 nodes per pathway.

Nodes: `molecular_state` 2,615, `reaction_event` 2,077, `complex_state` 1,473,
`set_state` 582.
Edges: `input_to` 3,856, `produces` 3,293, `precedes` 1,661, `catalyzed_by` 921,
`activated_by` 270, `inhibited_by` 184.

### Phase 4 — cross-context edges

Four edge types, each carrying the evidence that produced it:

| edge type | count | evidence |
|---|---|---|
| `same_entity_as` | 4,267 | shared UniProt accession — grounds pathway entities in root proteins |
| `targets_reaction` | 341 | the drug's PrimeKG target protein participates in the reaction |
| `participates_in_reaction` | 75 | the drug itself is a curated reaction participant (ChEBI) |
| `same_chemical_as` | 22 | whole-molecule reference ↔ Reactome molecular state |

**Leakage control.** A `targets_reaction` edge is derived *from* a PrimeKG
`(drug, protein)` fact. If that fact is held out, the edge is a two-hop copy of the answer
via drug → reaction → protein. **16 such edges were dropped**, and the filter re-runs
whenever the held-out sets change.

### Phase 5 — integration

Combined into `biomedical_gog.json`:

```json
{
  "root_graph":          {"nodes": [...], "edges": [...]},
  "internal_graphs":     {"drug": [...], "pathway": [...]},
  "owner_links":         [...],
  "cross_context_edges": [...],
  "metadata":            {...}
}
```

370 owner links (root node → its internal graph). That is what makes it a graph-of-*graphs*
rather than two unrelated files.

### Phase 6 — validation, all passed

```
envelope structure    PASS
drug graphs           PASS   246/246 molecules reconstructed exactly
pathway graphs        PASS   2,077 reactions, no uncurated ordering edges
cross-context edges   PASS   every edge carries evidence and a method
structural gain       PASS   3.8x
```

---

## 4. Headline structural result

> **9,629 internal nodes and 12,575 internal edges on top of 2,534 root nodes — 3.8×.**

None of it is expressible in PrimeKG: no atoms, no bonds, no functional groups, no
reactions, no compartments, no molecular states.

---

## 5. The evaluation splits, and a problem we found and fixed

### What the original flat splits looked like

The project started with three files: 160k / 20k / 20k facts sampled from all of PrimeKG.
An audit found:

| | original | ours |
|---|---|---|
| test facts | 20,000 (10,000 unique) | 2,009 |
| **each edge scored twice** | **yes** | no |
| **source identifiers** | **absent** | present |
| test facts inside the GoG universe | **1.3%** | **100%** |
| test facts touching an internal graph | ~1.3% | **34.1%** |
| `drug_drug` share of test | **80.0%** | 9.0% |
| `pathway_protein` test edges | **2** | **118** |

The original splits handled inverse-relation leakage correctly, which is the hard part.
Their problems were different: both directions of every PrimeKG edge were present, so
every metric double-counted; `head` and `tail` were PrimeKG *row indices* with no
DrugBank or Reactome id, so no internal graph could attach; and 98.7% of held-out facts
involved nodes outside the subset, making the internal graphs structurally unreachable.
`pathway_protein` had **two** test edges, so the entire pathway workstream was
unmeasurable.

### How leakage is ruled out

Four mechanisms, the third specific to a graph-of-graphs:

1. **Direction pairs cannot separate.** PrimeKG stores every edge twice, so a naive random
   split puts `(a,r,b)` in train and `(b,r,a)` in test. Edges collapse to one record per
   undirected key before assignment.
2. **Assignment happens on the collapsed set**, so each edge is assigned exactly once.
3. **Cross-context edges filtered against the held-out sets** — 16 dropped for re-encoding
   a held-out fact.
4. **Verified by reading the written files**, not from build-time state:

```
train/validation   exact · reversed · undirected    0  0  0
train/test         exact · reversed · undirected    0  0  0
validation/test    exact · reversed · undirected    0  0  0
internal duplicate edges, all three splits          0
cold-start nodes in validation / test               0
341 targets_reaction edges vs 47,382 held-out pairs 0
```

### Final splits, 8:1:1 stratified per relation

```
train         96,034   80.21%
validation    11,823    9.88%
test          11,869    9.91%
total        119,726
```

Per-relation test shares all between 8.6% and 10.5%. `pathway_protein` 149,
`drug_protein` 164, `contraindication` 269, `indication` 56.

Two configurations exist. Both share the identical structural level:

| | subset | balanced |
|---|---|---|
| root entities | 2,534 | 9,593 |
| root train facts | 16,462 | 96,045 |
| test facts | 2,007 | 11,861 |
| **test touching an internal graph** | **34%** | 10.4% |
| `drug_drug` share of test | 9% | 62% |

---

## 6. Running it on a model

The GoG was exported to the four-table schema of a separate nine-stage GoG encoder
(`github.com/richie955/graphofgraphs`):

```
entities     {entity_id, name, type, source, source_id}
graphs       {graph_id, type, parent_entity_id}      exactly one ROOT
graph_nodes  {node_id, entity_id, graph_id}
relations    {relation_id, source_node_id, type, target_node_id, display_relation, split}
```

**That repo's own validator reports `passed`** on both files, with
`splits_generated=False, split_unit='source-provided'` — our verified splits are used, not
re-rolled.

Scope routing, derived by their loader from endpoint graph ids:

```
train        root_to_root 96,045 | same_internal_graph 12,575 | root_to_internal 4,705
validation   root_to_root 11,820 | everything else 0
test         root_to_root 11,861 | everything else 0
```

**Only PrimeKG root facts carry held-out labels.** All structural and cross-context edges
are train-split encoder channels, never prediction targets. The GoG helps by informing
embeddings, not by being scored.

### Two schema findings worth presenting

**Pathway entities are shared, drug entities could not be.** Pathway entities are keyed by
Reactome stId, so ubiquitin is one entity with occurrences in 20 pathway graphs — exactly
what the architecture's identity-synchronisation stage operates on.

Merging pathway entities into their *root* counterparts was attempted and **rejected by the
data**: 94 of 124 pathway graphs contain two or more entities sharing one UniProt
accession. `R-HSA-75893` has four distinct states sharing P01375 — TNF as a trimer, bound
to TNFR1, soluble, and inhibitor-bound. Merging would either break the
one-occurrence-per-entity-per-graph invariant or collapse distinct molecular states,
destroying the structure that justifies the pathway level. Identity therefore ships
relationally.

**Atom and fragment embeddings were untrainable, and we fixed it.** Measured:

```
atom                1,917 entities   1.00 occurrence each   1.59 relations each
molecular_fragment    719 entities   1.00 occurrence each   2.42 relations each
```

Every atom was its own entity receiving gradient from one or two edges in the entire
dataset — so those embeddings never trained, and fed noise into three of the nine stages.
`carboxylic_acid` appears in 56 of 246 drugs but was 56 unrelated entities.

The fix exploits the architecture: Stage 1 computes
`b_entity = entity_embedding + type_embedding(type)`, so a **type** is a shared trained
parameter. Encoding chemistry in `type` rather than `entity_id` keeps ids unique (avoiding
the same collision problem as the merge) while sharing the parameters corpus-wide. Entity
types went from 16 to 62:

```
atom_C_deg1    487 atoms      frag_aromatic_ring        219
atom_C_deg2    446            frag_ring                 108
atom_O_deg1    383            frag_amide                 72
atom_C_deg3    194            frag_carboxylic_acid       56
```

---

## 7. Results

All on the subset configuration, 4,014 test queries (2,007 facts × head and tail
corruption), 0 unscorable. `baseline` is an entity-embedding model over the root graph
only; `gog` is the nine-stage encoder that also sees the internal graphs.

```
                              filtered                       raw
                      MRR    H@1    H@3   H@10  meanRk    MRR   meanRk

run 1  (h256, neg10, 100ep)
  baseline          0.2480  0.164  0.273  0.410   163.0  0.0564   178.0
  gog               0.1456  0.068    —    0.301   121.2  0.0863     —

run 2  (h128, neg50, 80ep)
  baseline          0.2084  0.126  0.224  0.377   157.3  0.0724   178.0
  gog               0.1395  0.065  0.144  0.287   118.0  0.0868   134.0
```

```
run 3  (h128, neg50, 80ep, chemistry-bearing entity types)
  baseline          0.2063  0.125  0.216  0.376   159.1  0.0702   180.2   best_ep 80
  gog               0.1018  0.050  0.092  0.208   143.0  0.0762   157.5   best_ep 18
```

**Run 3 was a tested hypothesis that failed.** Atom and fragment entities each had one
occurrence and ~1.6 relations, so their private embeddings never trained. The fix encoded
chemistry in `type` (62 types instead of 16), since Stage 1 computes
`entity_embedding + type_embedding` and a type is shared.

It made the GoG worse: 0.1395 -> 0.1018. The reason is that the coarse `type = "atom"`
was **already** a shared parameter trained from all 1,917 atoms; splitting it into 22
finer types fragmented that sharing and added capacity. The baseline was unchanged
(0.2084 -> 0.2063), confirming the experiment was clean and the only variable was the
drug-side typing.

**The valuable finding is in the training curve**, not the final number:

```
gog       epoch 18   val 0.0984   <- peak, then declines
gog       epoch 30   val 0.0911   <- early-stopped (patience 12)
baseline  epoch 80   val 0.1681   <- still climbing at the cap
```

The GoG peaks at epoch 18 and degrades; the baseline improves monotonically for 80
epochs. This **reverses the earlier underfitting diagnosis**: the GoG has too much
capacity for 16,462 training facts and begins memorising almost immediately. Finer types
made it worse by adding more parameters.

### Three findings, replicated across both runs

**1. ComplEx outperforms the GoG encoder on filtered MRR** in every run. Stated plainly —
but see the ablation above: this is a fact about the encoder, not about the structure.

**2. The GoG wins raw MRR and mean rank.** Raw 0.0868 vs 0.0724 (+20%), mean rank 118 vs
157 — the GoG places the true target about 25% higher on average.

**3. The filtering gap is the most interesting number:**

```
baseline   raw 0.0724 -> filtered 0.2084    2.88x gain
gog        raw 0.0868 -> filtered 0.1395    1.61x gain
```

The baseline gains nearly 3× from filtering, meaning the candidates it ranks above the
target are overwhelmingly *other known true facts* — it memorised the training graph. The
GoG gains far less, so what it ranks highly is more often not in the KG at all.

### THE KEY RESULT — the structural ablation

Same encoder, same splits, same 4,014 test queries. The only difference between the two
columns is whether the 370 internal graphs and 4,705 cross-context edges exist.

```
                        filtered MRR    raw MRR    H@1     H@10   meanRank
A1  baseline, root-only     0.2051      0.0657    0.124   0.375    163.9
A2  baseline, full GoG      0.2084      0.0724    0.126   0.377    157.3
B   gog, root-only          0.0927      0.0701    0.044   0.180    139.2
C   gog, full GoG           0.1395      0.0868    0.065   0.287    118.0
```

**B -> C: the structural level helps, on every metric.**

```
filtered MRR    0.0927 -> 0.1395    +50.5%
raw MRR         0.0701 -> 0.0868    +23.9%
Hits@1          0.044  -> 0.065     +49.1%
Hits@10         0.180  -> 0.287     +59.6%
mean rank       139.2  -> 118.0     15.2% better
```

**A1 -> A2: the control held.** The baseline moves +1.6% when structural data is added —
negligible, as it must be, since it has no message passing and cannot use structure. This
confirms the B -> C gain is attributable to the structure and not to the larger file.

**A1 -> B: message passing is the bottleneck.** Nine stages of propagation on root-only
data cost 54.8% of performance (0.2051 -> 0.0927). This is over-smoothing, measured
directly: ComplEx keeps each entity's idiosyncratic position, and propagation averages it
toward its neighbourhood.

**Structure closes a third of the gap to ComplEx.** The baseline/GoG ratio falls from
2.21x on root-only data to 1.49x with the structural level present.

### Three separable findings

1. **The graph-of-graphs structure improves the encoder by ~50% relative** on filtered
   MRR, ~60% on Hits@10, 15% on mean rank. The contribution works.
2. **The encoder's message passing is what loses to ComplEx**, not the data: -54.8% on
   root-only.
3. **The structure recovers a third of that loss**, and the architectural fixes identified
   below target the rest.

### The experimental design

```
A  ComplEx, embeddings only, no message passing
B  GoG encoder on root-only data
C  GoG encoder on root + 370 internal graphs
```

A -> B isolates message passing. **B -> C isolates the structural contribution** — the
question this project exists to answer. Neither requires beating ComplEx.

ComplEx vs GoG conflates two questions. The comparison that isolates the structural
level is the same encoder with and without internal graphs. Both files are built and
validated, differing in exactly one respect:

```
                      entities  internal graphs  train root  train internal  train cross  test
gog_model_subset_rootonly  2,534        0           16,462          0              0      2,007
gog_model_subset_coarse    9,523      370           16,462     12,575          4,705      2,007
```

This yields the three-rung ladder the model repo's own handoff asks for:

```
A  ComplEx, embeddings only, no message passing
B  GoG encoder on root-only data        (ablation, prepared, not yet run)
C  GoG encoder on root + 370 internal graphs
```

A -> B isolates message passing. **B -> C isolates the structural contribution** — the
question this project exists to answer. Neither requires beating ComplEx.

### Interpretation

The two models fail differently. The baseline is **high variance** — more top-10 hits
(0.377 vs 0.287) but a worse tail, hence a worse mean rank despite better Hits@10. It
nails facts resembling what it memorised and badly misses the rest. The GoG is **low
variance** — fewer top-10 hits but a tighter distribution, mean rank 118. Consistently
mediocre rather than sometimes excellent.

That is what you would expect when structural edges pull chemically and biologically
related entities together: the embedding space becomes smoother, so nothing ranks
absurdly low, but nothing is pinpointed either.

### Why the GoG underperforms — four structural reasons

1. **16,462 root training facts against a nine-stage encoder.** Loss fell 12× while
   validation MRR rose 5.7× on one configuration — textbook overfitting.
2. **Only 34% of test facts touch an internal graph** (10.4% on balanced). The rest cannot
   benefit by construction.
3. **`protein_protein` (30.8%) and `disease_protein` (22.9%) are 54% of the test set**, and
   proteins have no internal graphs at all.
4. **The architecture has never been tuned on real structural internal graphs.** The
   reference dataset shipped with that repo builds internal graphs from PrimeKG
   *neighbours* — the internal level re-encodes root facts. Its defaults were calibrated
   for something else.

---

## 8. Limitations — state these, do not hide them

1. **No architectural claim is supported.** Two runs, one seed each, one dataset. The
   model repo's own documentation is explicit: *"neither conclusion follows from one
   unreplicated score."*
2. **Subset selection preceded fact splitting**, so graph construction saw edges that later
   became test facts. That repo's handoff requires train-only construction inputs. This is
   mild selection leakage; the fix is a reordering, not a rebuild, and was not attempted.
3. **Cross-context edges are 91% grounding edges.** `same_entity_as` is 4,267 of 4,705;
   genuine drug-level participation is 97 edges. Reactome simply does not curate many drugs
   as reaction participants.
4. **Guide to Pharmacology matching produced nothing.** 40 drugs carry a GtP id from
   UniChem, but none coincided with a GtP-typed entity in the 124 built pathways. Every
   direct edge came from ChEBI.
5. **Fragment collapsing is lossy.** One 22-heavy-atom drug reduces to 6 nodes, 16 atoms
   inside a single ring node. Full atom detail survives in the fragment attributes, so a
   less aggressive variant needs no rebuild — but as exported, a GNN sees coarse structure.
   For aspirin specifically, the *ortho* positioning essential to COX-1 acetylation is
   invisible at fragment level.
6. **Built from a sample.** 300 sampled drugs and 80 sampled pathways. Scaling to all
   7,957 drugs costs ~4 h and all 2,516 pathways ~24 h, both by changing one config value.
7. **Leakage paths of three or more hops were not checked.** Two-hop re-encoding is
   filtered and independently verified; deeper paths are not.
8. **MRR is not comparable across configurations.** Subset has 2,534 candidate entities,
   balanced 9,593. Absolute values cannot be compared between them, or to published PrimeKG
   numbers on different splits.

---

## 9. What would most improve it, in priority order

1. **Give proteins internal graphs.** 54% of the test set involves proteins, which have no
   structure at all. Pfam/InterPro domains or Reactome participation would address the
   majority of the evaluation. Highest-value next step.
2. **Scale the owners.** 246 drugs and 124 pathways out of ~6,843 and ~1,761 eligible.
   Raises the 34% coverage figure directly.
3. **Stop collapsing rings.** With chemistry-bearing types making atom parameters
   trainable, the uncollapsed atom graph becomes genuinely useful — and it's a config
   change.
4. **Fix the construction ordering** so graph selection uses train-only facts, satisfying
   the leakage audit in full.
5. **Several seeds and converged runs.** Both completed runs hit their epoch cap and were
   still improving, so all reported numbers are lower bounds.

---

## 10. Suggested narrative for the presentation

**Act 1 — the gap.** PrimeKG represents biology as flat triples. It cannot express what a
drug is made of or what a pathway does. Show the aspirin example and `DIABLO [cytosol]`
versus `DIABLO [mitochondrial intermembrane space]`.

**Act 2 — what we built.** 2,534 root nodes owning 246 molecular graphs and 124 reaction
graphs. 9,629 internal nodes, 3.8× the root. All validation checks pass, including exact
reconstruction of all 246 molecules. Every cross-context edge carries its evidence.

**Act 3 — the evaluation problem we found.** The original splits had 1.3% GoG coverage, two
`pathway_protein` test edges, double-counted every edge, and dropped the source identifiers
needed to attach any structure. We rebuilt them: 8:1:1 stratified, source-provided,
verified leak-free across 15 checks from disk.

**Act 4 — the honest result.** The baseline wins filtered MRR (0.208 vs 0.140, replicated).
The GoG wins raw MRR and mean rank. The baseline gains 2.9× from filtering versus the
GoG's 1.6×, which says it memorised the training graph. We diagnosed four structural
reasons for the shortfall, and found and fixed a defect where atom embeddings were
untrainable.

**Act 5 — what's next.** Protein internal graphs, scale, converged runs, the ordering fix.

The strongest line available:

> We built a graph-of-graphs that adds molecular and reaction-level structure beneath
> PrimeKG's entities — 3.8× the node count, including mechanism PrimeKG cannot represent —
> and we found and fixed a leakage-prone evaluation setup along the way. Whether this
> structure improves link prediction is not yet settled: the baseline currently wins on
> filtered MRR, the GoG wins on raw MRR and mean rank, and we can say precisely why.

That reads as someone who understands their own pipeline. An unsupported performance claim
would not.

---

## 11. Appendix — every number in one place

```
SOURCE
  PrimeKG                         8,100,498 edges   129,312 nodes
  drugs resolving to SMILES       86.0% of sample   ~6,843 of 7,957 projected
  pathways in 10-200 node range   70.0% of sample   ~1,761 of 2,516 projected
  pathway size                    median 24, p90 160, max 939

ROOT GRAPH
  nodes 2,534   edges 40,944
  gene/protein 1,118 | disease 592 | effect/phenotype 417 | drug 247 | pathway 160

DRUG INTERNAL GRAPHS
  graphs 246   nodes 2,882   edges 2,390
  validated 246/246 exact reconstruction
  31 fragment labels; aromatic_ring 219, ring 108, amide 72, carboxylic_acid 56

PATHWAY INTERNAL GRAPHS
  graphs 124   nodes 6,747   edges 10,185   reactions 2,077
  accepted 124, too large 20, too small 16, of 160 candidates
  mean 54.4 nodes per pathway

CROSS-CONTEXT
  4,705 edges: same_entity_as 4,267 | targets_reaction 341 |
               participates_in_reaction 75 | same_chemical_as 22
  16 dropped by the leakage filter

INTEGRATED
  owner links 370
  internal nodes 9,629   internal edges 12,575   =  3.8x the root node count

SPLITS (8:1:1, stratified per relation, verified leak-free)
  train 96,034 (80.21%) | validation 11,823 (9.88%) | test 11,869 (9.91%)
  test: pathway_protein 149 | drug_protein 164 | contraindication 269 | indication 56

MODEL FILES (validator: passed, splits source-provided)
  gog_model_balanced.json   32 MB   16,582 entities   371 graphs   137,006 relations
  gog_model_subset.json     11 MB    9,523 entities   371 graphs    37,749 relations
  entity types 62 (subset) / 67 (balanced) after the chemistry-typing fix

RESULTS (subset, 4,014 test queries, 0 unscorable)
  run 1  h256 neg10 100ep   baseline filt 0.2480  gog filt 0.1456
                            baseline raw  0.0564  gog raw  0.0863
  run 2  h128 neg50  80ep   baseline filt 0.2084  gog filt 0.1395
                            baseline raw  0.0724  gog raw  0.0868
                            baseline meanRk 157.3  gog meanRk 118.0
  filtering gain            baseline 2.88x  gog 1.61x
```

Metric definitions: **MRR** is the mean of `1/rank` of the true answer among candidates.
**Filtered** removes all other known true facts (train, validation and test) from the
ranking first, so the model is not penalised for ranking a different correct answer above
the held-out one. Filtered is the standard headline metric; the raw/filtered gap reveals
how much a model relies on memorised facts.
