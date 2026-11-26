# Structural Graph-of-Graphs — data, pipeline and results

A graph-of-graphs over a PrimeKG subset where **drug nodes own molecular-structure
graphs** (PubChem → RDKit) and **pathway nodes own reaction graphs** (Reactome), plus
evidence-backed cross-context edges between the two levels.

Built to feed `gog_embedding/`. Every file here passes this repo's own
`data_pipeline.preprocessing.load_gog()` validator.

---

## The headline result

Same encoder, same splits, same 4,014 test queries. The only difference is whether the
370 internal graphs and 4,705 cross-context edges exist.

```
                        filtered MRR    raw MRR    H@1     H@10   meanRank
A1  baseline, root-only     0.2051      0.0657    0.124   0.375    163.9
A2  baseline, full GoG      0.2084      0.0724    0.126   0.377    157.3
B   gog, root-only          0.0927      0.0701    0.044   0.180    139.2
C   gog, full GoG           0.1395      0.0868    0.065   0.287    118.0
```

**B → C — adding the structural level improves the encoder on every metric:**
filtered MRR **+50.5%**, Hits@1 **+49.1%**, Hits@10 **+59.6%**, mean rank **15.2% better**.

**A1 → A2 — the control held.** The entity-only baseline moves just +1.6% when structural
data is added, as it must, since it has no message passing and cannot use structure. This
confirms the B → C gain comes from the structure and not from the larger file.

**A1 → B — message passing is the bottleneck.** Nine stages on root-only data cost 54.8%
(0.2051 → 0.0927). That is over-smoothing: ComplEx keeps each entity's position sharp,
propagation averages it toward its neighbourhood.

ComplEx still leads overall, but the structural level closes a third of the gap — the
baseline/GoG ratio falls from 2.21× to 1.49×.

Settings for all four: subset data, `hidden_dim 128`, `negatives 50`, 80 epochs,
patience 15, seed 17.

---

## What was built

```
root graph              2,534 nodes    40,944 edges
drug molecular graphs     246          2,882 nodes    2,390 edges
pathway reaction graphs   124          6,747 nodes   10,185 edges   2,077 reactions
owner links               370
cross-context edges     4,705          16 dropped by the leakage filter

structural gain         9,629 internal nodes = 3.8x the root node count
```

None of it is expressible in PrimeKG: no atoms, no bonds, no functional groups, no
reactions, no compartments, no molecular states.

All Phase 6 checks pass, including **exact bond-set reconstruction for 246/246 molecules**.

---

## What is here

**Code and documentation only.** No data files — the pipeline regenerates all of them
from public sources, and the JSON artifacts ran to 122 MB.

```
README.md                    this file
docs/
  01_PROJECT_HANDOFF.md      the full write-up: problem, methodology, every number,
                             limitations, suggested presentation narrative. Start here.
  02_BUILD_RESULTS.md        build statistics and leakage verification
  03_SCHEMA_DECISIONS.md     the model-input schema and why each decision was made
  04_HOW_TO_RUN_ON_COLAB.md  training sequence
pipeline/
  00..12_*.py                the thirteen build steps, in order
  src/gog/*.py               the library they import
  config.yaml                all thresholds; no script hardcodes a limit
  requirements.txt
  optional_patch_stage1_skip.py   a proposed encoder fix, not part of the build
```

## Reproducing the data

```bash
pip install -r structural_gog/pipeline/requirements.txt
```

Download PrimeKG `kg.csv` from the Harvard Dataverse record `doi:10.7910/DVN/IXA7BM`
and place it at `structural_gog/pipeline/data/raw/kg.csv`, then:

```bash
cd structural_gog/pipeline
python 01_coverage_check.py        # ~9 min   resolve ids, measure coverage
python 02_build_root_graph.py      # ~10 min  subset + root graph
python 03_make_splits.py           # seconds  leakage-free stratified splits
python 04_build_drug_graphs.py     # seconds  246 molecular graphs
python 05_build_pathway_graphs.py  # ~56 min  124 reaction graphs
python 06_build_cross_edges.py     # ~2 min   cross-context edges + leakage filter
python 07_integrate.py             # seconds  biomedical_gog.json
python 08_validate.py              # seconds  Phase 6 validation
python 10_make_hybrid_splits.py    # ~2 min   absorb extra training facts
python 11_make_balanced_splits.py  # seconds  8:1:1, verified leak-free
python 12_export_to_model.py --splits subset --coarse-types   # the model input file
```

Every HTTP response is cached under `data/cache/`, so a re-run takes minutes rather than
the ~80 minutes of the first build, and it is reproducible offline afterwards.

### The files step 12 produces

| flags | file | role |
|---|---|---|
| `--splits subset --coarse-types` | `gog_model_subset_coarse.json` | **the main file.** Root graph + 370 internal graphs + cross edges. Gave GoG 0.1395 |
| `--splits subset --coarse-types --no-internal` | `gog_model_subset_rootonly.json` | identical root graph, **structure removed**. Gave GoG 0.0927. This pair is the ablation |
| `--splits subset` | `gog_model_subset.json` | 62 chemistry-bearing entity types. A tested hypothesis that failed: 0.1018 |
| `--splits balanced --coarse-types` | `gog_model_balanced.json` | 96,045 root training facts instead of 16,462; 10.4% structural coverage vs 34% |

Then train:

```bash
python -m training.train <file> --model compare \
    --epochs 80 --patience 15 --hidden-dim 128 --negatives 50 --device cuda \
    --output-dir runs/with_structure
```

Splits travel inside the file on `relations[].split`, so the loader reports
`splits_generated=False, split_unit='source-provided'` and does not re-roll them.

Two evaluation slices are also produced by step 11:
`balanced_{validation,test}_gog_slice.json`, the 1,238 and 1,240 held-out facts with an
owner on an endpoint. **Report metrics on these alongside the full split** — the full
split measures the model, the slice isolates the structural contribution.

## Known issues in this repo's code, found while integrating

1. **`baselines/rgcn.py` imports `SCOPES`** from `data_pipeline.preprocessing`, which does
   not export it. `python -m training.train` fails on import at `2eed6d1`. One-line fix:
   re-export `SCOPES` and `SPLITS` from `validate_gog_data` in that package's `__init__`.
2. **`--model compare` runs only `baseline` and `gog`**, not `rgcn`, so the documented
   four-rung ladder is not what `compare` delivers.
3. **R-GCN needs PyTorch Geometric**, which is not in any requirements file. It raises a
   clear `RuntimeError`, but a run with no other output looks like a silent hang.
4. **The Stage-1 state never reaches the decoder.** `GoGEncoder.forward` overwrites `z`
   every round, so the sharp pre-propagation embedding is destroyed. The entity-only
   baseline is therefore **not** a special case of the encoder, which is why the encoder
   cannot match it. A gated residual would fix this — see
   `pipeline/optional_patch_stage1_skip.py`. It applies cleanly and trains, but has not
   been evaluated at full scale.

## Limitations of this data

1. **Subset selection preceded fact splitting**, so graph construction saw edges that
   later became test facts. `GOG_BUILDER_TEAMMATE_HANDOFF.md` §3.4 requires train-only
   construction inputs. Mild selection leakage; the fix is a reordering, not a rebuild.
2. **Structure reaches only 34% of held-out facts**, and that ceiling is PrimeKG's
   relation mix, not a build gap — 54% of test is `protein_protein` or `disease_protein`,
   where neither endpoint carries structure. Owner coverage inside the subset is already
   100% of drugs and 78% of pathways. **Protein internal graphs are the highest-value
   next step.**
3. **Cross-context edges are 91% grounding** (`same_entity_as` 4,267). Genuine drug-level
   participation is 97 edges — Reactome curates few drugs as reaction participants.
4. **Fragment collapsing is lossy.** A 22-heavy-atom drug reduces to 6 nodes. Full atom
   detail survives in the fragment attributes, so an uncollapsed variant needs no rebuild.
5. **Single seed per configuration**, and the runs hit their epoch cap, so numbers are
   lower bounds.
6. **MRR is not comparable across configurations** — subset has 2,534 candidate entities,
   balanced 9,593.
