"""Regenerate the self-contained Colab split notebook from the tested module."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
module_source = (ROOT / "data_pipeline" / "preprocessing" / "prepare_primekg_splits.py").read_text(encoding="utf-8")


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"name": "prepare_primekg_splits.ipynb"},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": [
        cell("markdown", """# Prepare a reproducible PrimeKG split package

Run this notebook in Google Colab. It selects original PrimeKG facts, makes a pair-grouped train/validation/test split, validates coverage, and downloads one ZIP. It **does not** build GoG contexts or call an LLM. The GoG builder must filter `selected_facts.csv` to `split == 'train'` before computing any neighborhoods or prompts.

Defaults target 200,000 unique source facts and 25 supported owners of each type. Exact 80/10/10 ratios are not guaranteed because entity pairs stay together. The input is `kg.csv` from PrimeKG; the notebook first checks Google Drive and then offers upload.
"""),
        cell("code", """# Colab normally has these packages. Install only when missing.
import importlib.util, subprocess, sys
missing = [name for name in ('numpy', 'pandas') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
"""),
        cell("code", module_source),
        cell("code", """# Parameters: edit this cell before running the build.
TARGET_FACTS = 200_000
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
OWNERS_PER_TYPE = 25  # each of drug, gene/protein, pathway
MIN_ENTITIES = 10_000
MAX_ENTITIES = 50_000
NEIGHBORS_PER_FRONTIER_NODE = 24
MIN_OWNER_TRAIN_GROUPS = 5

PARAMS = SplitParameters(
    target_facts=TARGET_FACTS, train_ratio=TRAIN_RATIO,
    validation_ratio=VALIDATION_RATIO, test_ratio=TEST_RATIO,
    seed=SEED, owners_per_type=OWNERS_PER_TYPE,
    min_entities=MIN_ENTITIES, max_entities=MAX_ENTITIES,
    neighbors_per_frontier_node=NEIGHBORS_PER_FRONTIER_NODE,
    min_owner_train_groups=MIN_OWNER_TRAIN_GROUPS,
)
PARAMS.validate()
print(PARAMS)
"""),
        cell("code", """# Load kg.csv from Google Drive, or upload it to this runtime.
from pathlib import Path
INPUT_CSV = Path('/content/drive/MyDrive/kg.csv')
if not INPUT_CSV.is_file():
    from google.colab import drive, files
    drive.mount('/content/drive')
    if not INPUT_CSV.is_file():
        print('Select your PrimeKG kg.csv file for upload.')
        uploaded = files.upload()
        if 'kg.csv' not in uploaded:
            raise FileNotFoundError('Upload a file named kg.csv, or place it in MyDrive/kg.csv')
        INPUT_CSV = Path('/content/kg.csv')
print('Input:', INPUT_CSV)
"""),
        cell("code", """# Build, validate, summarize, and download the package.
from google.colab import files
OUTPUT_ZIP = Path('/content/primekg_split_package.zip')
report = build_split_package(INPUT_CSV, OUTPUT_ZIP, PARAMS)
print('Output:', OUTPUT_ZIP)
print('Selected facts:', sum(report['fact_counts_by_split'].values()))
print('Split counts:', report['fact_counts_by_split'])
print('Actual ratios:', report['actual_ratios'])
print('Owners:', report['owner_counts_by_type'])
print('Source SHA-256:', report['source_sha256'])
print('Selected fact set SHA-256:', report['selected_fact_set_sha256'])
files.download(str(OUTPUT_ZIP))
"""),
        cell("markdown", """## Builder handoff

Extract the ZIP. Give `selected_entities.csv` to the builder as the ROOT entity universe. Give it only the `split == 'train'` rows from `selected_facts.csv` for adjacency, degrees, neighborhood retrieval, LLM prompts, and internal graph creation. Validation and test rows are labels for later evaluation. Do not create new cross-context semantic facts by copying ROOT relations.
"""),
    ],
}

destination = ROOT / "notebooks" / "prepare_primekg_splits.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)

# Keep the adapted LLM builder executable as separate Colab cells. Its raw
# source is retained beside this generator for review and ordinary diffing.
builder_source = (ROOT / "data_pipeline" / "construction" / "build_gog_from_split.py").read_text(encoding="utf-8")
builder_parts = [part for part in re.split(r"(?=^# CELL \d+:)", builder_source, flags=re.MULTILINE)
                 if part.strip()]
builder_notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"name": "primekg_llm_gog_split_safe.ipynb"},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": [cell("markdown", """# LLM-assisted GoG builder using a frozen split

First run `prepare_primekg_splits.ipynb` and provide its `primekg_split_package.zip`. This notebook uses only **training facts** for owner selection, neighborhoods, prompts, internal memberships, and internal edges. ROOT includes the frozen train/validation/test facts as labeled source records so Stage 0 can evaluate held-out ROOT links. Derived internal edges are marked as training-only computational channels and are not supervised targets. No contextual cross-graph facts are synthesized.
"""), *[cell("code", part) for part in builder_parts]],
}
builder_destination = ROOT / "notebooks" / "primekg_llm_gog_split_safe.ipynb"
builder_destination.write_text(json.dumps(builder_notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(builder_destination)
