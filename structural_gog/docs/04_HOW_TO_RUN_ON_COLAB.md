# Running the structural GoG on the model — Colab

Two files are ready in `data/out/`. Both pass the model repo's own
`load_gog()` validator, and both carry **source-provided** splits
(`splits_generated=False`), so the loader uses your verified 8:1:1 rather than
re-rolling at seed 42.

| file | size | entities | graphs | relations (train / val / test) |
|---|---|---|---|---|
| `gog_model_balanced.json` | 32 MB | 16,582 | 371 | 113,325 / 11,820 / 11,861 |
| `gog_model_subset.json` | 11 MB | 9,523 | 371 | 33,742 / 2,000 / 2,007 |

Both contain the same structural level: 246 drug molecular graphs and 124 pathway
reaction graphs, 370 internal graphs in total.

---

## Step 1 — put the file in Drive

Upload `data/out/gog_model_balanced.json` to the top level of your Google Drive.
32 MB, so it takes a moment but survives runtime restarts — unlike `files.upload()`,
which you would repeat on every fresh runtime.

## Step 2 — Colab, GPU runtime

Runtime → Change runtime type → T4 GPU.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!git clone https://github.com/richie955/graphofgraphs.git /content/gog
%cd /content/gog
```

**One fix needed before anything runs.** `baselines/rgcn.py` imports `SCOPES` from
`data_pipeline.preprocessing`, but that package's `__init__.py` does not export it,
so `training.train` fails on import at the current HEAD:

```python
%%writefile /content/gog/data_pipeline/preprocessing/__init__.py
"""Data preparation modules for GoG."""
from .validate_gog_data import (
    GoGValidationError, PreprocessedGoG, SCOPES, SPLITS, load_gog, preprocess_data,
)

__all__ = ["GoGValidationError", "PreprocessedGoG", "SCOPES", "SPLITS",
           "load_gog", "preprocess_data"]
```

Worth telling whoever owns the repo — it's a one-line export, not a design problem.

## Step 3 — validate before training

```python
import sys; sys.path.insert(0, '/content/gog')
from data_pipeline.preprocessing import load_gog

DATA = '/content/drive/MyDrive/gog_model_balanced.json'
d = load_gog(DATA)
print(d.report['validation'], d.report['counts'])
print('splits_generated:', d.report['splits_generated'])
print(d.report['scope_counts_by_split'])
```

Expect `passed`, and `splits_generated: False`. If that second one ever prints
`True`, the split labels were not read and the run is not comparable to your audit.

## Step 4 — the comparison run

This is the evaluation ladder from `docs/09` — entity baseline, R-GCN, then GoG, all
on identical splits, candidates and decoder.

```python
!python -m training.train "$DATA" --model compare \
    --epochs 150 --patience 20 --hidden-dim 256 --negatives 10 \
    --device cuda --output-dir /content/drive/MyDrive/gog_runs/balanced
```

Writing output to Drive means the reports survive a disconnect.

For a second seed — their doc is explicit that one unreplicated score proves nothing:

```python
!python -m training.train "$DATA" --model compare --seed 23 \
    --epochs 150 --patience 20 --device cuda \
    --output-dir /content/drive/MyDrive/gog_runs/balanced_seed23
```

## Step 5 — read the results

```python
import json
r = json.load(open('/content/drive/MyDrive/gog_runs/balanced/comparison_report.json'))
for kind in ('baseline', 'rgcn', 'gog'):
    if kind in r:
        t = r[kind]['test']['mrr']
        print(f"{kind:9s} filtered MRR {t['filtered']['micro']:.4f}  "
              f"raw {t['raw']['micro']:.4f}")
```

---

## Timing

The subset file runs 2 epochs at hidden 64 in 3 seconds on a laptop CPU. Scaling to
hidden 256, 150 epochs and 6x the facts, expect roughly 20–60 minutes on a T4 for the
balanced comparison. Start it and leave it.

## What to report, and what not to claim

Report per-relation and per-scope MRR and Hits@1/3/10, candidate counts, and seed
variation. Their doc is direct about the interpretation:

> Interpret B>A as evidence that context information helps, and C>B as evidence that
> the GoG encoder uses the same context better; neither conclusion follows from one
> unreplicated score.

Two caveats that belong in the talk:

1. **Held-out facts are PrimeKG root facts only.** All 12,575 structural edges and
   4,705 cross-context edges are train-split encoder channels, never prediction
   targets. The GoG helps by informing embeddings, not by being scored.
2. **Subset selection preceded fact splitting**, so graph construction saw edges
   that later became test facts. Handoff §3.4 requires train-only construction
   inputs. Mild selection leakage; the fix is reordering, not rebuilding.

`drug_drug` is 62% of the balanced test set, so lead with per-relation numbers rather
than the micro average.
