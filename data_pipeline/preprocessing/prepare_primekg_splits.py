"""Create a reproducible source-fact split package from raw PrimeKG kg.csv.

This module deliberately creates no GoG contexts or contextual copies. The
training-only GoG builder consumes its selected_facts.csv after filtering split.
"""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OWNER_TYPES = ("drug", "gene/protein", "pathway")
FACT_COLUMNS = ("x_index", "x_type", "x_name", "y_index", "y_type", "y_name", "relation")
FACT_KEY = ("x_index", "relation", "y_index")


@dataclass(frozen=True)
class SplitParameters:
    target_facts: int = 200_000
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    owners_per_type: int = 25
    min_entities: int = 10_000
    max_entities: int = 50_000
    neighbors_per_frontier_node: int = 24
    min_owner_train_groups: int = 5

    def validate(self) -> None:
        if self.target_facts < 1 or self.owners_per_type < 1 or self.min_owner_train_groups < 1:
            raise ValueError("Fact, owner, and training-support counts must be positive")
        if self.min_entities < 1 or self.max_entities < self.min_entities:
            raise ValueError("Require 1 <= min_entities <= max_entities")
        if self.neighbors_per_frontier_node < 1:
            raise ValueError("neighbors_per_frontier_node must be positive")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Train, validation, and test ratios must be positive and sum to one")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fact_id(row: tuple[str, str, str]) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _read_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(FACT_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"PrimeKG CSV is missing columns: {sorted(missing)}")
    usecols = [*FACT_COLUMNS, *(["display_relation"] if "display_relation" in header else [])]
    facts = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, low_memory=False)
    facts.insert(0, "source_row", np.arange(1, len(facts) + 1, dtype=np.int64))
    for col in ("x_index", "y_index", "x_type", "y_type", "relation"):
        facts[col] = facts[col].str.strip()
    if "display_relation" not in facts:
        facts["display_relation"] = ""
    invalid = (facts[["x_index", "y_index", "x_type", "y_type", "relation"]] == "").any(axis=1)
    invalid |= facts["x_index"] == facts["y_index"]
    invalid_count = int(invalid.sum())
    facts = facts.loc[~invalid].copy()

    duplicate_columns = ["source_row", *FACT_KEY, "display_relation"]
    duplicates = facts.loc[facts.duplicated(list(FACT_KEY), keep="first"), duplicate_columns].copy()
    facts = facts.drop_duplicates(list(FACT_KEY), keep="first").reset_index(drop=True)
    facts.insert(1, "fact_id", [_stable_fact_id(values) for values in facts[list(FACT_KEY)].itertuples(index=False, name=None)])
    duplicates.insert(1, "fact_id", [_stable_fact_id(values) for values in duplicates[list(FACT_KEY)].itertuples(index=False, name=None)])

    left = facts[["x_index", "x_type", "x_name"]].rename(columns={"x_index": "entity_id", "x_type": "type", "x_name": "name"})
    right = facts[["y_index", "y_type", "y_name"]].rename(columns={"y_index": "entity_id", "y_type": "type", "y_name": "name"})
    entities = pd.concat((left, right), ignore_index=True)
    conflicts = entities.groupby("entity_id", sort=False)["type"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting entity types for {int((conflicts > 1).sum())} IDs")
    entities = entities.drop_duplicates("entity_id", keep="first").sort_values("entity_id").reset_index(drop=True)
    source_report = {"input_rows": int(len(facts) + len(duplicates) + invalid_count),
                     "invalid_or_self_loop_rows": invalid_count,
                     "exact_duplicate_rows": int(len(duplicates)),
                     "unique_directed_facts": int(len(facts))}
    return facts, entities, duplicates, source_report


def _pair_key(facts: pd.DataFrame) -> pd.Series:
    left = facts["x_index"].to_numpy(dtype=str)
    right = facts["y_index"].to_numpy(dtype=str)
    first = np.where(left < right, left, right)
    second = np.where(left < right, right, left)
    return pd.Series(first + "\x1f" + second, index=facts.index)


def _pick_owner_seeds(entities: pd.DataFrame, facts: pd.DataFrame, p: SplitParameters) -> list[str]:
    degree = Counter(facts["x_index"]) + Counter(facts["y_index"])
    rng = random.Random(p.seed)
    seeds = []
    for owner_type in OWNER_TYPES:
        candidates = [eid for eid in entities.loc[entities["type"] == owner_type, "entity_id"]
                      if degree[eid] >= p.min_owner_train_groups]
        if len(candidates) < p.owners_per_type:
            raise ValueError(f"Only {len(candidates)} source-supported {owner_type} owners; need {p.owners_per_type}")
        candidates.sort(key=lambda eid: (degree[eid], eid))
        # Sample across low, medium, and high degree instead of only hubs.
        bins = [candidates[(i * len(candidates) + 2) // 3:((i + 1) * len(candidates) + 2) // 3]
                for i in range(3)]
        for i in range(p.owners_per_type):
            pool = bins[i % 3] or next(bucket for bucket in bins if bucket)
            seeds.append(pool.pop(rng.randrange(len(pool))))
    return seeds


def _select_entity_neighborhood(facts: pd.DataFrame, seeds: list[str], p: SplitParameters) -> tuple[set[str], dict]:
    ids = pd.Index(pd.unique(pd.concat((facts["x_index"], facts["y_index"]), ignore_index=True))).sort_values()
    id_to_index = {eid: i for i, eid in enumerate(ids)}
    src = facts["x_index"].map(id_to_index).to_numpy(dtype=np.int32)
    dst = facts["y_index"].map(id_to_index).to_numpy(dtype=np.int32)
    relation = facts["relation"].to_numpy()
    source_type = facts["x_type"].to_numpy()
    target_type = facts["y_type"].to_numpy()
    ends = np.concatenate((src, dst))
    edge_ids = np.tile(np.arange(len(facts), dtype=np.int32), 2)
    order = np.argsort(ends, kind="stable")
    edge_ids = edge_ids[order]
    ptr = np.concatenate(([0], np.cumsum(np.bincount(ends, minlength=len(ids)), dtype=np.int64)))
    selected = np.zeros(len(ids), dtype=bool)
    queue = deque()
    for eid in seeds:
        index = id_to_index[eid]
        if not selected[index]:
            selected[index] = True
            queue.append(index)
    rng = np.random.default_rng(p.seed)
    selected_count = len(queue)
    induced_count = 0
    last_check = 0
    while queue:
        if selected_count >= p.min_entities and selected_count - last_check >= 500:
            induced_count = int(np.count_nonzero(selected[src] & selected[dst]))
            last_check = selected_count
            if induced_count >= p.target_facts:
                break
        if selected_count >= p.max_entities:
            break
        node = queue.popleft()
        adjacent = edge_ids[ptr[node]:ptr[node + 1]]
        # Round-robin relation/type buckets; random order inside each bucket.
        buckets = defaultdict(list)
        for edge_id in adjacent:
            other_type = target_type[edge_id] if src[edge_id] == node else source_type[edge_id]
            buckets[(relation[edge_id], other_type)].append(edge_id)
        bucket_keys = sorted(buckets)
        for key in bucket_keys:
            buckets[key] = list(rng.permutation(buckets[key]))
        adjacent = []
        while any(buckets.values()) and len(adjacent) < p.neighbors_per_frontier_node * 4:
            for key in bucket_keys:
                if buckets[key]:
                    adjacent.append(buckets[key].pop())
        added = 0
        for edge_id in adjacent:
            other = dst[edge_id] if src[edge_id] == node else src[edge_id]
            if selected[other]:
                continue
            selected[other] = True
            queue.append(int(other))
            selected_count += 1
            added += 1
            if added >= p.neighbors_per_frontier_node or selected_count >= p.max_entities:
                break
    induced_count = int(np.count_nonzero(selected[src] & selected[dst]))
    if induced_count < p.target_facts:
        raise ValueError(f"Neighborhood has only {induced_count:,} facts after {selected_count:,} entities; "
                         "increase max_entities or lower target_facts")
    return set(ids[selected].tolist()), {"explored_entities": selected_count,
                                        "induced_fact_pool": induced_count}


def _select_fact_groups(pool: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    pool = pool.copy()
    pool["pair_key"] = _pair_key(pool)
    groups = {key: rows.index.to_list() for key, rows in pool.groupby("pair_key", sort=True)}
    if len(pool) <= target:
        return pool
    relation_counts = pool["relation"].value_counts().to_dict()
    strata = defaultdict(list)
    for key, positions in groups.items():
        relation = min((pool.at[pos, "relation"] for pos in positions),
                       key=lambda name: (relation_counts[name], name))
        strata[relation].append(key)
    rng = random.Random(seed)
    for keys in strata.values():
        rng.shuffle(keys)
    chosen = set()
    count = 0
    for relation in sorted(strata):
        quota = round(target * relation_counts[relation] / len(pool))
        used = 0
        for key in strata[relation]:
            size = len(groups[key])
            if used >= quota or count + size > target:
                continue
            chosen.add(key)
            used += size
            count += size
    remaining = [key for key in groups if key not in chosen]
    rng.shuffle(remaining)
    for key in remaining:
        size = len(groups[key])
        if count + size <= target:
            chosen.add(key)
            count += size
    return pool.loc[pool["pair_key"].isin(chosen)].copy()


def _split_and_choose_owners(facts: pd.DataFrame, entities: pd.DataFrame,
                             p: SplitParameters) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    facts = facts.copy()
    groups = {key: rows.index.to_list() for key, rows in facts.groupby("pair_key", sort=True)}
    rng = random.Random(p.seed + 1)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    reserved = set()
    by_entity = defaultdict(list)
    by_relation = defaultdict(list)
    for key in group_keys:
        rows = groups[key]
        first = facts.loc[rows[0]]
        by_entity[first["x_index"]].append(key)
        by_entity[first["y_index"]].append(key)
        for relation in facts.loc[rows, "relation"].unique():
            by_relation[relation].append(key)
    # A training fact for every evaluated entity and relation.
    for entity_id in sorted(by_entity):
        reserved.add(by_entity[entity_id][0])
    for relation in sorted(by_relation):
        reserved.add(by_relation[relation][0])

    entity_types = entities.set_index("entity_id")["type"].to_dict()
    owners = []
    for owner_type in OWNER_TYPES:
        candidates = [eid for eid in sorted(by_entity) if entity_types[eid] == owner_type
                      and len(by_entity[eid]) >= p.min_owner_train_groups]
        if len(candidates) < p.owners_per_type:
            raise ValueError(f"Only {len(candidates)} eligible {owner_type} owners in subset")
        candidates.sort(key=lambda eid: (len(by_entity[eid]), eid))
        bins = [candidates[(i * len(candidates) + 2) // 3:((i + 1) * len(candidates) + 2) // 3]
                for i in range(3)]
        for i in range(p.owners_per_type):
            pool = bins[i % 3] or next(bucket for bucket in bins if bucket)
            eid = pool.pop(rng.randrange(len(pool)))
            owners.append((eid, owner_type))
            reserved.update(by_entity[eid][:p.min_owner_train_groups])

    desired = {"validation": round(len(facts) * p.validation_ratio),
               "test": round(len(facts) * p.test_ratio)}
    remaining = [key for key in group_keys if key not in reserved]
    if sum(len(groups[key]) for key in remaining) < sum(desired.values()):
        raise ValueError("Coverage and owner reservations leave too few facts for validation/test")
    assignments = {key: "train" for key in reserved}
    counts = Counter({"train": sum(len(groups[key]) for key in reserved)})
    # Pair-group allocation aims at fact-count targets; exact ratios may be impossible.
    for key in remaining:
        size = len(groups[key])
        deficits = {split: max(0, desired[split] - counts[split]) for split in ("validation", "test")}
        if sum(deficits.values()) <= 0:
            split = "train"
        else:
            split = max(("validation", "test"), key=lambda name: (deficits[name] / max(desired[name], 1), name))
            if deficits[split] < size and counts[split] > 0:
                other = "test" if split == "validation" else "validation"
                if deficits[other] >= size:
                    split = other
        assignments[key] = split
        counts[split] += size
    facts["split"] = facts["pair_key"].map(assignments)
    train = facts.loc[facts["split"] == "train"]
    training_degree = Counter(train["x_index"]) + Counter(train["y_index"])
    owner_rows = []
    for entity_id, owner_type in owners:
        pair_count = int(train.loc[(train["x_index"] == entity_id) | (train["y_index"] == entity_id), "pair_key"].nunique())
        if pair_count < p.min_owner_train_groups:
            raise AssertionError("Selected owner lost required training support")
        owner_rows.append({"entity_id": entity_id, "type": owner_type,
                           "training_fact_count": int(training_degree[entity_id]),
                           "training_pair_count": pair_count})
    return facts, pd.DataFrame(owner_rows).sort_values(["type", "entity_id"]), {
        "reserved_training_pair_groups": len(reserved),
        "target_fact_counts": {"train": round(len(facts) * p.train_ratio), **desired},
    }


def _validate_package(facts: pd.DataFrame, entities: pd.DataFrame,
                      owners: pd.DataFrame, p: SplitParameters) -> dict:
    if facts["fact_id"].duplicated().any():
        raise AssertionError("Fact IDs overlap")
    if facts.groupby("pair_key")["split"].nunique().max() != 1:
        raise AssertionError("An entity pair crosses splits")
    if set(facts["split"]) != {"train", "validation", "test"}:
        raise ValueError("All three splits must contain facts")
    train = facts.loc[facts["split"] == "train"]
    if not (set(facts["x_index"]) | set(facts["y_index"])).issubset(set(train["x_index"]) | set(train["y_index"])):
        raise AssertionError("An evaluation entity has no training fact")
    if not set(facts["relation"]).issubset(set(train["relation"])):
        raise AssertionError("An evaluation relation has no training fact")
    owner_counts = owners["type"].value_counts().to_dict()
    if any(owner_counts.get(t, 0) != p.owners_per_type for t in OWNER_TYPES):
        raise AssertionError("Owner count mismatch")
    split_counts = facts["split"].value_counts().to_dict()
    relation_counts = pd.crosstab(facts["relation"], facts["split"]).to_dict(orient="index")
    entity_type_counts = entities["type"].value_counts().to_dict()
    return {"validation": "passed", "fact_counts_by_split": split_counts,
            "actual_ratios": {split: split_counts.get(split, 0) / len(facts)
                              for split in ("train", "validation", "test")},
            "relation_counts_by_split": relation_counts,
            "entity_counts_by_type": entity_type_counts,
            "owner_counts_by_type": owner_counts,
            "pair_groups": int(facts["pair_key"].nunique()),
            "training_only_builder_rule": "Filter selected_facts.csv to split == train before all adjacency, degree, retrieval, or LLM operations"}


def build_split_package(source_csv: str | Path, output_zip: str | Path,
                        params: SplitParameters = SplitParameters()) -> dict:
    """Build and verify a ZIP containing source facts, split labels, and audit metadata."""
    params.validate()
    source_csv, output_zip = Path(source_csv), Path(output_zip)
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    source_hash = _sha256_file(source_csv)
    facts, entities, duplicates, source_report = _read_source(source_csv)
    seeds = _pick_owner_seeds(entities, facts, params)
    selected_entity_ids, selection_report = _select_entity_neighborhood(facts, seeds, params)
    induced = facts.loc[facts["x_index"].isin(selected_entity_ids) &
                        facts["y_index"].isin(selected_entity_ids)].copy()
    selected = _select_fact_groups(induced, params.target_facts, params.seed)
    selected, owners, split_report = _split_and_choose_owners(selected, entities, params)
    selected_ids = set(selected["x_index"]) | set(selected["y_index"])
    selected_entities = entities.loc[entities["entity_id"].isin(selected_ids)].copy()
    audit = _validate_package(selected, selected_entities, owners, params)
    audit.update({"source": source_report, "selection": selection_report,
                  "split": split_report, "source_sha256": source_hash,
                  "selected_fact_set_sha256": hashlib.sha256("\n".join(sorted(selected["fact_id"])).encode()).hexdigest(),
                  "split_manifest_sha256": hashlib.sha256("\n".join(
                      f"{fid}:{split}" for fid, split in sorted(zip(selected["fact_id"], selected["split"])))
                      .encode()).hexdigest()})
    manifest = selected[["fact_id", "pair_key", "split"]].sort_values("fact_id")
    exported = selected.drop(columns="pair_key").sort_values("fact_id")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in (("selected_facts.csv", exported), ("split_manifest.csv", manifest),
                            ("selected_entities.csv", selected_entities), ("owners.csv", owners),
                            ("duplicate_source_rows.csv", duplicates)):
            archive.writestr(name, frame.to_csv(index=False, lineterminator="\n"))
        archive.writestr("parameters.json", json.dumps(asdict(params), indent=2, sort_keys=True))
        archive.writestr("audit_report.json", json.dumps(audit, indent=2, sort_keys=True))
    return {"output_zip": str(output_zip), **audit}
