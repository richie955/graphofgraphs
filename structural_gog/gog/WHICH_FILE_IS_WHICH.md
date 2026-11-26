# Which file is which

Eleven JSON files from the structural graph-of-graphs build. Two kinds: four
**model-input** files you feed to `python -m training.train`, and seven **split and
support** files that are human-readable or intermediate.

Code and documentation live in the `graphofgraphs` repo under `structural_gog/`.

---

## Feed these to the model

All four carry their splits inside, on `relations[].split`, so the loader reports
`splits_generated=False, split_unit='source-provided'` and does **not** re-roll them.
All four pass `data_pipeline.preprocessing.load_gog()`.

| file | size | entities | graphs | relations | what it is |
|---|---|---|---|---|---|
| **`gog_model_subset_coarse.json`** | 10 MB | 9,523 | 371 | 37,749 | **The main file.** Root graph + 370 internal graphs + 4,705 cross-context edges. Simple entity types (11). Gave **GoG 0.1395** |
| **`gog_model_subset_rootonly.json`** | 4.5 MB | 2,534 | 1 | 20,469 | **The ablation.** Identical root graph, **all structure removed**. Gave **GoG 0.0927**. Compare against the file above — that pair is the experiment |
| `gog_model_subset.json` | 10 MB | 9,523 | 371 | 37,749 | Same as the main file but with 62 chemistry-bearing entity types instead of 11. **A tested hypothesis that failed:** GoG 0.1018 |
| `gog_model_balanced.json` | 31 MB | 16,582 | 371 | 137,006 | Scaled up — 96,045 root training facts instead of 16,462. Lower absolute MRR (9,593-entity candidate universe) and 10.4% structural coverage vs 34% |

```bash
python -m training.train gog_model_subset_coarse.json --model compare \
    --epochs 80 --patience 15 --hidden-dim 128 --negatives 50 --device cuda \
    --output-dir runs/with_structure
```

### The result these produced

```
                        filtered MRR    raw MRR    H@1     H@10   meanRank
A1  baseline, root-only     0.2051      0.0657    0.124   0.375    163.9
A2  baseline, full GoG      0.2084      0.0724    0.126   0.377    157.3
B   gog, root-only          0.0927      0.0701    0.044   0.180    139.2
C   gog, full GoG           0.1395      0.0868    0.065   0.287    118.0
```

**B → C — the structural level improves the encoder on every metric:** filtered MRR
+50.5%, Hits@1 +49.1%, Hits@10 +59.6%, mean rank 15.2% better.

**A1 → A2 — the control held.** The entity-only baseline moves +1.6% when structure is
added, as it must, having no message passing to use it with. That is what makes the
B → C gain attributable to the structure rather than to the larger file.

**A1 → B — message passing costs 54.8%** on root-only data. Over-smoothing.

---

## Split files — human-readable, not model input

These carry full names, types and source identifiers. Useful for inspection, slides and
auditing. The model reads the four files above instead.

| file | size | facts | what it is |
|---|---|---|---|
| `balanced_train_facts.json` | 55 MB | 96,034 | training split, 80.2% |
| `balanced_validation_facts.json` | 6.8 MB | 11,823 | validation split, 9.9% |
| `balanced_test_facts.json` | 6.7 MB | 11,869 | test split, 9.9% |
| `balanced_validation_gog_slice.json` | 748 KB | 1,238 | **only** the held-out validation facts with an owner on an endpoint |
| `balanced_test_gog_slice.json` | 745 KB | 1,240 | **only** the held-out test facts with an owner on an endpoint |

**Use the two slice files when reporting.** The full test split measures the model; the
slice isolates what the structural level contributes. 62% of the full balanced test set
is `drug_drug`, which pathway reaction graphs say nothing about.

### Leakage

The splits are 8:1:1, stratified per relation, and verified from disk across 15 checks —
exact, reverse-direction and undirected overlap between every pair of splits, internal
duplicates, and cold-start endpoints. All zero. Both directions of each PrimeKG edge are
kept in the same split and only one record is emitted, so no metric double-counts.

---

## Intermediate files

| file | size | what it is |
|---|---|---|
| `hybrid_train_facts.json` | 61 MB | superseded. The training set before the 8:1:1 rebalance — 115,714 facts at 96.6/1.7/1.7. Kept only for provenance |
| `hybrid_root_nodes.json` | 1.8 MB | the 9,593-node embedding table for the balanced configuration, with type counts |

---

## Quick reference

- **Running the real experiment?** `gog_model_subset_coarse.json` and
  `gog_model_subset_rootonly.json`.
- **Reporting numbers?** `balanced_test_gog_slice.json` alongside the full test split.
- **Want the bigger benchmark?** `gog_model_balanced.json`, but its MRR is not comparable
  to the subset files — 9,593 candidate entities versus 2,534.
- **`gog_model_subset.json`** exists only to reproduce the failed typing experiment.

Not in this folder: `biomedical_gog.json`, the 20 MB construction artifact with full
provenance — atom indices, internal bonds, cross-edge evidence. Regenerate it with
`structural_gog/pipeline/07_integrate.py`.
