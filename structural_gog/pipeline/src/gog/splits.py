"""Loading and auditing the flat train/val/test fact files.

Record shape produced by the split generator:

    {fact_id, head, head_name, head_type, relation, display_relation,
     tail, tail_name, tail_type, split, in_gog_root_universe}

`head` and `tail` are PrimeKG **x_index / y_index** values (row indices), not
source identifiers. There is no x_id / y_id field, so a drug here cannot be
mapped to DrugBank and a pathway cannot be mapped to Reactome without going back
to kg.csv. `index_to_source_id()` is the bridge; `audit()` reports what is
missing before any of it is attempted.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import REPO_ROOT

DEFAULT_FILES = {
    "train": "train-rich.json",
    "validation": "validation_data.json",
    "test": "test_data.json",
}


def load_split(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(path) as fh:
        doc = json.load(fh)
    return doc.get("metadata", {}), doc.get("facts", [])


def load_all(files: dict[str, str] | None = None, root: Path | None = None):
    files = files or DEFAULT_FILES
    root = root or REPO_ROOT
    out: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name, fname in files.items():
        p = Path(root) / fname
        if not p.exists():
            raise FileNotFoundError(f"missing split file: {p}")
        meta[name], out[name] = load_split(p)
    return meta, out


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def directed_key(f: dict) -> tuple[str, str, str]:
    return (f["head"], f["relation"], f["tail"])


def reverse_key(f: dict) -> tuple[str, str, str]:
    return (f["tail"], f["relation"], f["head"])


def undirected_key(f: dict) -> tuple[str, str, str]:
    a, b = sorted((f["head"], f["tail"]))
    return (a, f["relation"], b)


def unique_edges(facts: Iterable[dict]) -> set[tuple[str, str, str]]:
    """Collapses the two stored directions of one PrimeKG edge into one key."""
    return {undirected_key(f) for f in facts}


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


@dataclass
class SplitAudit:
    counts: dict[str, int] = field(default_factory=dict)
    unique_edge_counts: dict[str, int] = field(default_factory=dict)
    duplication_factor: dict[str, float] = field(default_factory=dict)
    cross_split_exact: dict[str, int] = field(default_factory=dict)
    cross_split_reverse: dict[str, int] = field(default_factory=dict)
    cross_split_undirected: dict[str, int] = field(default_factory=dict)
    relation_share: dict[str, list[tuple[str, int, float]]] = field(default_factory=dict)
    target_relation_edges: dict[str, dict[str, int]] = field(default_factory=dict)
    gog_universe: dict[str, int] = field(default_factory=dict)
    owner_coverage: dict[str, Any] = field(default_factory=dict)
    has_source_ids: bool = False
    problems: list[str] = field(default_factory=list)


TARGET_RELATIONS = (
    "indication", "contraindication", "drug_protein", "drug_drug",
    "pathway_protein", "pathway_pathway", "disease_protein",
)

ID_FIELDS = ("head_id", "tail_id", "x_id", "y_id", "head_source_id", "tail_source_id")


def audit(splits: dict[str, list[dict]], *, min_test_edges: int = 1000) -> SplitAudit:
    a = SplitAudit()
    sample = next(iter(splits.values()))[:1000]
    present = {k for f in sample for k in f}
    a.has_source_ids = any(k in present for k in ID_FIELDS)
    if not a.has_source_ids:
        a.problems.append(
            "BLOCKING: no source-identifier field (head_id/x_id/...). `head` and `tail` "
            "are PrimeKG row indices, so drugs carry no DrugBank id and pathways no "
            "Reactome stId. Internal graphs cannot be attached from these files alone."
        )

    for name, facts in splits.items():
        a.counts[name] = len(facts)
        uniq = unique_edges(facts)
        a.unique_edge_counts[name] = len(uniq)
        a.duplication_factor[name] = round(len(facts) / max(1, len(uniq)), 3)
        rel = Counter(f["relation"] for f in facts)
        a.relation_share[name] = [
            (r, c, round(c / max(1, len(facts)), 4)) for r, c in rel.most_common(12)
        ]
        a.gog_universe[name] = sum(1 for f in facts if f.get("in_gog_root_universe"))

    for rel in TARGET_RELATIONS:
        a.target_relation_edges[rel] = {
            name: len({undirected_key(f) for f in facts if f["relation"] == rel})
            for name, facts in splits.items()
        }

    # Leakage: every held-out split against train.
    if "train" in splits:
        tr_dir = {directed_key(f) for f in splits["train"]}
        tr_uni = unique_edges(splits["train"])
        for name, facts in splits.items():
            if name == "train":
                continue
            a.cross_split_exact[name] = sum(1 for f in facts if directed_key(f) in tr_dir)
            a.cross_split_reverse[name] = sum(1 for f in facts if reverse_key(f) in tr_dir)
            a.cross_split_undirected[name] = sum(1 for f in facts if undirected_key(f) in tr_uni)

    # Owner-node reach: can the internal graphs influence the held-out metric at all?
    for name, facts in splits.items():
        drug_facts = sum(1 for f in facts if "drug" in (f["head_type"], f["tail_type"]))
        pw_facts = sum(1 for f in facts if "pathway" in (f["head_type"], f["tail_type"]))
        a.owner_coverage[name] = {
            "facts_touching_drug": drug_facts,
            "facts_touching_pathway": pw_facts,
            "drug_share": round(drug_facts / max(1, len(facts)), 4),
            "pathway_share": round(pw_facts / max(1, len(facts)), 4),
        }

    # Verdicts.
    for name in splits:
        if name == "train":
            continue
        if a.duplication_factor.get(name, 1) > 1.5:
            a.problems.append(
                f"{name}: every edge is stored in both directions (factor "
                f"{a.duplication_factor[name]}), so reported metrics count each edge twice. "
                f"Score one direction per edge, or deduplicate with undirected_key()."
            )
        cov = a.owner_coverage[name]
        if cov["facts_touching_pathway"] < 50:
            a.problems.append(
                f"{name}: only {cov['facts_touching_pathway']} facts touch a pathway node, "
                f"so the pathway internal graphs have no measurable effect on this split."
            )
        if a.gog_universe.get(name, 0) < min_test_edges:
            a.problems.append(
                f"{name}: only {a.gog_universe[name]} facts are inside the GoG root "
                f"universe, so any GoG-vs-baseline difference rests on that many facts."
            )
        top = a.relation_share[name][0] if a.relation_share.get(name) else None
        if top and top[2] > 0.5:
            a.problems.append(
                f"{name}: relation '{top[0]}' is {top[2]:.0%} of the split, so the headline "
                f"metric is essentially that one relation. Report per-relation metrics."
            )
    return a


def index_to_source_id(kg_csv_path: str | Path) -> dict[str, dict[str, str]]:
    """PrimeKG row index -> {'id','type','name','source'}, the missing bridge.

    Rebuilds what the split files dropped. Needed before any internal graph can
    be attached to a fact in those files.
    """
    import pandas as pd

    frames = []
    for side in ("x", "y"):
        frames.append(
            pd.read_csv(
                kg_csv_path,
                usecols=[f"{side}_index", f"{side}_id", f"{side}_type", f"{side}_name", f"{side}_source"],
                dtype=str,
                low_memory=False,
            ).rename(columns={
                f"{side}_index": "index", f"{side}_id": "id", f"{side}_type": "type",
                f"{side}_name": "name", f"{side}_source": "source",
            })
        )
    nodes = pd.concat(frames, ignore_index=True).drop_duplicates(subset="index")
    return {
        str(r.index_): {"id": str(r.id), "type": str(r.type), "name": str(r.name), "source": str(r.source)}
        for r in nodes.rename(columns={"index": "index_"}).itertuples()
    }
