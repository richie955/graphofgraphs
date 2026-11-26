# Graph of Graphs

This repository contains a staged GoG data pipeline, the GoG embedding model, training code, and experiment artifacts.

## Where things live

| Path | Purpose |
| --- | --- |
| `data_pipeline/preprocessing/` | PrimeKG fact splitting and GoG input validation |
| `data_pipeline/construction/` | Build GoG structures from the frozen training split |
| `data/raw/` | Original source datasets |
| `data/splits/` | Train, validation, and test facts |
| `data/processed/` | Constructed GoG model inputs |
| `gog_embedding/` | GoG embedding model stages |
| `training/` | Training entry points and shared loops |
| `baselines/` | Baseline models |
| `tests/` | Unit and integration tests |
| `notebooks/` | Colab workflows and notebook generator |
| `docs/` | Data contracts and model architecture notes |
| `runs/` | Experiment reports |

## Data pipeline files

- `preprocessing/prepare_primekg_splits.py` selects a PrimeKG subset, deduplicates facts, and assigns leakage-safe train/validation/test splits.
- `preprocessing/validate_gog_data.py` validates and indexes constructed GoG JSON for the model.
- `construction/build_gog_from_split.py` is the source for the Colab builder that creates GoG data from the frozen training split.

The flow is **split facts → construct GoG → validate model input → train/embed**. Run commands from the repository root. See [COLAB.md](COLAB.md) for the Colab workflow.

## Structural embedding experiments

The model now supports legacy and structural encoder profiles, baseline checkpoint
transfer, ROOT-only supervision, and fixed-budget baseline/R-GCN/GoG comparisons.
See [the embedding experiment guide](docs/structural_embedding_experiments.md) for
CPU smoke checks, full comparison commands, ablations, and report definitions.
This workflow reads existing GoG exports without changing their construction or splits.
