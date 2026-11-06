# Synthetic GoG Fixture

`synthetic_gog_fixture.json` is a small, fictional, portable input fixture for developing the GoG pipeline before selecting a real dataset. It uses only JSON and can be loaded directly in local Python or Google Colab. It is a plumbing/debugging fixture, not a benchmark or biomedical knowledge source.

## Contents

- 13 globally identified entities across five types: `Protein`, `Drug`, `Disease`, `Pathway`, and `Process`.
- One ROOT and two one-level internal graphs, each with a distinct owner that also has exactly one ROOT GraphNode.
- 14 GraphNodes, with no repeated `(entity_id, graph_id)` pair. `MEK` intentionally occurs in both ROOT and MAPK contexts to exercise shared identity with contextual states.
- Six directed relation predicates with simple head/tail type constraints.
- All supported relation scopes: ROOT→ROOT, same-internal, ROOT→internal, internal→ROOT, and internal↔different-internal. Different-internal relations exercise both directions across the two internal contexts.
- Explicit train, validation, and test facts. Held-out facts `F003` and `F007` must be absent from every encoder adjacency and its generated reverse computational channels.
- No ownership facts, duplicated facts, or generated semantic inverse predicates.

## Expected edge routing

Classify each fact using its endpoint GraphNode graph IDs, then route training facts only:

| Pipeline route | Fixture examples | All-fact count |
|---|---|---:|
| Stage 6 ROOT→ROOT | `F001`–`F003` | 3 |
| Stage 2 same internal graph | `F004`–`F007`, `F013`–`F014` | 6 |
| Stage 3 ROOT→internal | `F008`, `F010`, `F012` | 3 |
| Stage 3 internal→ROOT | `F009` | 1 |
| Stage 3 internal↔different internal | `F011`, `F015` | 2 |

The `expected_scope_counts_all_facts` field in the JSON is a quick integrity check and should match the table above. The fixture includes 15 facts total; there are six same-internal facts and two inter-internal facts in opposite directions.

## Using in Colab

Upload `synthetic_gog_fixture.json` to the Colab session, then load and inspect it:

```python
import json
from pathlib import Path

data = json.loads(Path("synthetic_gog_fixture.json").read_text())
print(len(data["entities"]), len(data["graph_nodes"]), len(data["facts"]))
```

To run the Stage 0 preprocessing code in a notebook, place `gog_preprocessing.py` beside the JSON file (or add its folder to `sys.path`):

```python
from gog_preprocessing import load_gog
from gog_model import GoGEncoder

prepared = load_gog("synthetic_gog_fixture.json")
print(prepared.report)
print(prepared.edge_lists["same_internal_graph"])

model = GoGEncoder.from_preprocessed(prepared, hidden_dim=256)
states = model(prepared)
print(states.z_occurrence.shape)  # [number of GraphNodes, 256]
print(states.h_graph.shape)       # [number of internal graphs, 256]
```

Locally, the same implementation can be run as a CLI:

```text
python gog_preprocessing.py docs/synthetic_gog_fixture.json
python gog_preprocessing.py docs/synthetic_gog_fixture.json --output /content/preprocessed.json
```

The preprocessing module uses only the Python standard library. Model stages are kept in separate files: `gog_model/stage1.py` through `gog_model/stage9.py`. `gog_model/__init__.py` is the simple entry point that joins them. The encoder requires PyTorch 2.x; Colab users can select a compatible PyTorch runtime. `python -m unittest -v` runs preprocessing and model checks.

To check that link training and evaluation run, use a short fixture run:

```text
python train_stage9.py docs/synthetic_gog_fixture.json --model compare --output-dir runs/fixture-check --epochs 10 --patience 4 --negatives 2 --hidden-dim 32 --device cpu
```

The default command trains only the entity-and-type baseline. Add `--model compare` to train both that baseline and GoG. Comparison saves both checkpoints plus `comparison_report.json`; baseline-only mode saves `baseline_best.pt` and `baseline_report.json`. Fixture scores are only a software check, not evidence of model quality. For a real dataset in the Stage 0 JSON format, pass its path instead of the fixture path and use the dataset's prescribed split.

Alternatively, keep the file beside the notebook in Drive or fetch it from the project repository once the project is hosted. The fixture has no local-path assumptions or extra dependencies.

## Notes for implementation

- Treat split labels in this tiny fixture as fixed test cases, not statistically meaningful splits.
- The same fact must route to exactly one scope based on endpoint graphs.
- Build message-passing edges from `split == "train"` only. Reverse channels are created only from those training facts inside their stage.
- Candidate entities must have the required type and a GraphNode in the required endpoint graph context.
- Owner mappings pair `G_MAPK`→`E_MAPK` and `G_PI3K`→`E_PI3K`; they never become edges.
- The fixture intentionally has no empty or singleton internal graph; empty/singleton behavior is covered by stage-level unit cases or a future fixture variant.
