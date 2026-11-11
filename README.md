# Graph of Graphs

This repository contains the Graph of Graphs model, PrimeKG preprocessing and training code, notebooks, test fixtures, and experiment reports.

## Directory map

| Path | Contents |
| --- | --- |
| `gog_model/` | Model stages and decoder |
| `gog_data/` | Graph preprocessing and PrimeKG split packaging |
| `training/` | Training and covered pilot evaluation |
| `baselines/` | Baseline models |
| `tests/` | Unit and integration tests |
| `notebooks/` | Colab and data preparation notebooks |
| `facts/` | PrimeKG train, validation, and test source facts |
| `docs/` | Design notes and graph fixtures |
| `runs/` | JSON reports from experiments; model checkpoints are ignored |

`prime_kg_gog_from_training.json` is the training graph used with the source facts in `facts/`. Run commands from the repository root so Python can import the packages above. See [COLAB.md](COLAB.md) for the Colab workflow.

The reports in `runs/primekg-covered-pilot-interrupted-2026-11-12/` came from a local pilot stopped before its final summary was written. The completed covered pilot reports are in `runs/primekg-covered-pilot/`.
