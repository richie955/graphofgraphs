"""Dependency-free Stage 0 loader and validator for Graph-of-Graphs data."""

from __future__ import annotations

import argparse
import json
import random
from functools import cached_property
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCOPES = ("root_to_root", "same_internal_graph", "root_to_internal", "internal_to_root", "internal_to_different_internal")
SPLITS = {"train", "validation", "test"}


class GoGValidationError(ValueError):
    """Raised when source data violates the Stage 0 contract."""


@dataclass(frozen=True)
class RoutedFact:
    fact_id: str
    relation_type_id: str
    relation_index: int
    source_node_id: str
    target_node_id: str
    source_occurrence_index: int
    target_occurrence_index: int
    source_graph_id: str
    target_graph_id: str
    scope: str
    split: str
    direction: str = "FORWARD"


@dataclass
class PreprocessedGoG:
    entities: list[dict[str, Any]]
    graphs: list[dict[str, Any]]
    graph_nodes: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    relation_schema: dict[str, dict[str, list[str]]]
    entity_id_to_index: dict[str, int]
    graph_id_to_index: dict[str, int]
    node_id_to_index: dict[str, int]
    relation_id_to_index: dict[str, int]
    type_id_to_index: dict[str, int]
    type_index_by_entity: list[int]
    entity_index_by_occurrence: list[int]
    graph_index_by_occurrence: list[int]
    root_occurrence_by_entity: list[int]
    has_root_occurrence: list[bool]
    owner_entity_by_graph: list[int]
    has_owner: list[bool]
    owner_root_by_graph: list[int]
    has_owner_root: list[bool]
    owned_graph_by_entity: list[int]
    has_owned_graph: list[bool]
    edge_lists: dict[str, list[RoutedFact]]
    report: dict[str, Any]

    @cached_property
    def graph_node_by_id(self) -> dict[str, dict[str, Any]]:
        return {row["node_id"]: row for row in self.graph_nodes}

    @cached_property
    def candidate_node_indices_by_graph(self) -> dict[str, list[int]]:
        """Occurrence indices grouped once by context for Stage 9 lookups."""
        grouped: dict[str, list[int]] = {}
        for index, node in enumerate(self.graph_nodes):
            grouped.setdefault(node["graph_id"], []).append(index)
        return grouped

    @cached_property
    def candidate_node_indices_by_graph_type(self) -> dict[tuple[str, str], list[int]]:
        """Occurrence indices grouped by context and entity type."""
        entity_types = {entity["entity_id"]: entity.get("type_id") for entity in self.entities}
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, node in enumerate(self.graph_nodes):
            key = (node["graph_id"], entity_types.get(node["entity_id"], "UNK"))
            grouped.setdefault(key, []).append(index)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation, including routed facts."""
        from dataclasses import asdict

        result = dict(self.__dict__)
        # Cached lookup indexes are derived runtime helpers, not part of the
        # serialized preprocessing artifact.
        for key in ("graph_node_by_id", "candidate_node_indices_by_graph",
                    "candidate_node_indices_by_graph_type"):
            result.pop(key, None)
        result["edge_lists"] = {key: [asdict(edge) for edge in edges] for key, edges in self.edge_lists.items()}
        return result


def _id_map(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {row[key]: i for i, row in enumerate(rows)}


def _duplicates(rows: list[dict[str, Any]], key: str, label: str, errors: list[str]) -> None:
    counts = Counter(row.get(key) for row in rows)
    repeated = sorted((str(value) for value, count in counts.items() if count > 1))
    if repeated:
        errors.append(f"duplicate {label} IDs: {', '.join(repeated[:8])}")


def classify_scope(source_graph_id: str, target_graph_id: str, root_graph_id: str) -> str:
    if source_graph_id == root_graph_id and target_graph_id == root_graph_id:
        return "root_to_root"
    if source_graph_id == target_graph_id:
        return "same_internal_graph"
    if source_graph_id == root_graph_id:
        return "root_to_internal"
    if target_graph_id == root_graph_id:
        return "internal_to_root"
    return "internal_to_different_internal"


def _normalize_source_data(data: dict[str, Any], split_seed: int = 42) -> dict[str, Any]:
    """Map the checked-in GoG JSON field names to the loader's in-memory names.

    The source file remains unchanged. ``type`` on a relation is the broad
    predicate family; ``display_relation`` is retained as part of the model
    relation ID so labels such as target/enzyme are not collapsed together.
    """
    if "facts" in data:
        return data
    if not all(key in data for key in ("entities", "graphs", "graph_nodes", "relations")):
        return data

    entities = [dict(row, type_id=row.get("type_id", row.get("type", "UNK")))
                for row in data["entities"]]
    graphs = []
    for row in data["graphs"]:
        graph = dict(row)
        graph["kind"] = graph.get("kind", graph.get("type"))
        if graph["kind"] == "internal":
            graph["owner_entity_id"] = graph.get("owner_entity_id", graph.get("parent_entity_id"))
        graphs.append(graph)

    facts = []
    for row in data["relations"]:
        fact = dict(row)
        broad_type = str(row.get("type", ""))
        display_type = row.get("display_relation")
        # Keep the source's more specific label when available.
        relation_type = f"{broad_type}::{display_type}" if display_type else broad_type
        fact["fact_id"] = row.get("fact_id", row.get("relation_id"))
        fact["relation_type_id"] = row.get("relation_type_id", relation_type)
        facts.append(fact)

    # Split by canonical entity pair, across every graph context and relation
    # label. The same semantic link may appear in ROOT and several internal
    # graphs with different GraphNode IDs; occurrence-pair grouping misses it.
    pair_groups: dict[tuple[str, str], list[int]] = {}
    entity_by_node = {node.get("node_id"): node.get("entity_id")
                      for node in data["graph_nodes"]}
    for i, fact in enumerate(facts):
        split_value = fact.get("split")
        if split_value not in SPLITS and split_value not in (None, ""):
            raise GoGValidationError(
                f"fact {fact.get('fact_id')!r} has invalid split {split_value!r}; "
                "expected train, validation, or test"
            )
        source, target = fact.get("source_node_id"), fact.get("target_node_id")
        source_entity = entity_by_node.get(source)
        target_entity = entity_by_node.get(target)
        pair = (tuple(sorted((str(source_entity), str(target_entity))))
                if source_entity is not None and target_entity is not None
                else ("__invalid__", str(i)))
        pair_groups.setdefault(pair, []).append(i)

    group_split: dict[tuple[str, str], str] = {}
    unassigned_by_relation: dict[str, list[tuple[str, str]]] = {}
    for pair, indexes in pair_groups.items():
        existing = {facts[i].get("split") for i in indexes if facts[i].get("split") in SPLITS}
        if len(existing) > 1:
            ids = ", ".join(str(facts[i].get("fact_id")) for i in indexes)
            raise GoGValidationError(f"facts for one entity pair have conflicting splits: {ids}")
        if existing:
            group_split[pair] = next(iter(existing))
        else:
            relation_key = min(str(facts[i].get("relation_type_id", "")) for i in indexes)
            unassigned_by_relation.setdefault(relation_key, []).append(pair)

    rng = random.Random(split_seed)
    generated_splits = False
    for relation_key in sorted(unassigned_by_relation):
        pairs = unassigned_by_relation[relation_key]
        rng.shuffle(pairs)
        n = len(pairs)
        if n < 3:
            labels = ["train"] * n
        else:
            n_validation = max(1, round(n * 0.1))
            n_test = max(1, round(n * 0.1))
            if n_validation + n_test >= n:
                n_validation = n_test = 1
            labels = (["train"] * (n - n_validation - n_test)
                      + ["validation"] * n_validation + ["test"] * n_test)
        for pair, label in zip(pairs, labels):
            group_split[pair] = label
            generated_splits = True

    for pair, indexes in pair_groups.items():
        label = group_split[pair]
        for i in indexes:
            if facts[i].get("split") not in SPLITS:
                facts[i]["split"] = label

    normalized = dict(data)
    normalized["entities"] = entities
    normalized["graphs"] = graphs
    normalized["facts"] = facts
    normalized.setdefault("relation_schema", {})
    normalized["dataset_name"] = data.get("dataset_name", data.get("metadata", {}).get("dataset", "unnamed_dataset"))
    normalized["split_seed"] = split_seed
    normalized["splits_generated"] = generated_splits
    return normalized


def preprocess_data(data: dict[str, Any], split_seed: int = 42) -> PreprocessedGoG:
    """Validate raw GoG records and create stable mappings and train edge lists."""
    data = _normalize_source_data(data, split_seed)
    required_tables = ("entities", "graphs", "graph_nodes", "facts", "relation_schema")
    for name in required_tables:
        if name not in data:
            raise GoGValidationError(f"missing required top-level field: {name}")
    entities = sorted(data["entities"], key=lambda x: x.get("entity_id", ""))
    graphs = sorted(data["graphs"], key=lambda x: x.get("graph_id", ""))
    nodes = sorted(data["graph_nodes"], key=lambda x: x.get("node_id", ""))
    facts = sorted(data["facts"], key=lambda x: x.get("fact_id", ""))
    schema = data["relation_schema"]
    errors: list[str] = []

    for rows, key, label in ((entities, "entity_id", "entity"), (graphs, "graph_id", "graph"),
                             (nodes, "node_id", "GraphNode"), (facts, "fact_id", "fact")):
        _duplicates(rows, key, label, errors)
        for i, row in enumerate(rows):
            if not row.get(key):
                errors.append(f"{label} row {i} has empty {key}")

    entity_map, graph_map, node_map = (_id_map(entities, "entity_id"), _id_map(graphs, "graph_id"), _id_map(nodes, "node_id"))
    relation_ids = sorted(set(schema) | {fact.get("relation_type_id") for fact in facts if fact.get("relation_type_id")})
    relation_map = {rid: i for i, rid in enumerate(relation_ids)}
    observed_types = {entity.get("type_id") for entity in entities if entity.get("type_id")}
    type_ids = sorted(observed_types - {"UNK"}) + ["UNK"]
    type_map = {type_id: i for i, type_id in enumerate(type_ids)}
    roots = [g["graph_id"] for g in graphs if g.get("kind") == "root"]
    if len(roots) != 1:
        errors.append(f"expected exactly one ROOT graph; found {len(roots)}")
    root_id = roots[0] if len(roots) == 1 else data.get("root_graph_id")
    if root_id not in graph_map:
        errors.append(f"root_graph_id {root_id!r} does not resolve to a graph")
    for graph in graphs:
        if graph.get("kind") not in {"root", "internal"}:
            errors.append(f"graph {graph.get('graph_id')!r} has invalid kind {graph.get('kind')!r}")
        if graph.get("kind") == "root" and graph.get("owner_entity_id") is not None:
            errors.append(f"ROOT graph {graph['graph_id']!r} must not have an owner")

    seen_pairs: set[tuple[str, str]] = set()
    for node in nodes:
        eid, gid = node.get("entity_id"), node.get("graph_id")
        if eid not in entity_map:
            errors.append(f"GraphNode {node.get('node_id')!r} references unknown entity {eid!r}")
        if gid not in graph_map:
            errors.append(f"GraphNode {node.get('node_id')!r} references unknown graph {gid!r}")
        pair = (eid, gid)
        if pair in seen_pairs:
            errors.append(f"duplicate entity/graph occurrence pair {pair!r}")
        seen_pairs.add(pair)

    internal_graphs = [g for g in graphs if g.get("kind") == "internal"]
    owners: dict[str, str] = {}
    root_nodes_by_entity: dict[str, str] = {}
    if root_id in graph_map:
        for node in nodes:
            if node.get("graph_id") == root_id:
                root_nodes_by_entity[node.get("entity_id")] = node.get("node_id")
    for graph in internal_graphs:
        owner = graph.get("owner_entity_id")
        if owner not in entity_map:
            errors.append(f"internal graph {graph['graph_id']!r} has unknown or missing owner {owner!r}")
            continue
        if owner in owners:
            errors.append(f"entity {owner!r} owns multiple internal graphs")
        owners[owner] = graph["graph_id"]
        if owner not in root_nodes_by_entity:
            errors.append(f"owner {owner!r} of graph {graph['graph_id']!r} has no ROOT GraphNode")

    node_by_id = {node["node_id"]: node for node in nodes}
    scope_by_fact: dict[str, str] = {}
    split_by_entity_pair: dict[tuple[str, str], str] = {}
    conflicting_pairs: set[tuple[str, str]] = set()
    for fact in facts:
        fid = fact.get("fact_id")
        split = fact.get("split")
        if split not in SPLITS:
            errors.append(f"fact {fid!r} has invalid split {split!r}; expected train, validation, or test")
        sid, tid, rid = fact.get("source_node_id"), fact.get("target_node_id"), fact.get("relation_type_id")
        if sid not in node_by_id or tid not in node_by_id:
            errors.append(f"fact {fid!r} has unresolved endpoint(s): source={sid!r}, target={tid!r}")
            continue
        if rid not in relation_map:
            errors.append(f"fact {fid!r} references unknown relation type {rid!r}")
            continue
        source, target = node_by_id[sid], node_by_id[tid]
        entity_pair = tuple(sorted((source["entity_id"], target["entity_id"])))
        prior_split = split_by_entity_pair.setdefault(entity_pair, split)
        if split in SPLITS and prior_split != split and entity_pair not in conflicting_pairs:
            errors.append(f"canonical entity pair {entity_pair!r} appears in multiple splits")
            conflicting_pairs.add(entity_pair)
        source_entity = entities[entity_map[source["entity_id"]]]
        target_entity = entities[entity_map[target["entity_id"]]]
        rel = schema.get(rid, {})
        # Endpoint type rules are optional. When supplied, validate them; when
        # omitted, the dataset's own relation labels remain the only constraint.
        if rel.get("head_types") and source_entity.get("type_id") not in rel["head_types"]:
            errors.append(f"fact {fid!r} head type {source_entity.get('type_id')!r} is invalid for relation {rid!r}")
        if rel.get("tail_types") and target_entity.get("type_id") not in rel["tail_types"]:
            errors.append(f"fact {fid!r} tail type {target_entity.get('type_id')!r} is invalid for relation {rid!r}")
        if source["graph_id"] not in graph_map or target["graph_id"] not in graph_map:
            continue
        scope_by_fact[fid] = classify_scope(source["graph_id"], target["graph_id"], root_id)

    if errors:
        raise GoGValidationError("GoG data validation failed:\n- " + "\n- ".join(errors))

    occurrence_entity = [entity_map[n["entity_id"]] for n in nodes]
    occurrence_graph = [graph_map[n["graph_id"]] for n in nodes]
    root_occurrence = [-1] * len(entities)
    for node in nodes:
        if node["graph_id"] == root_id:
            root_occurrence[entity_map[node["entity_id"]]] = node_map[node["node_id"]]
    has_root = [i >= 0 for i in root_occurrence]
    internal_graph_list = sorted(internal_graphs, key=lambda x: x["graph_id"])
    owner_entity, owner_root, owned_graph = [], [], [-1] * len(entities)
    for graph in internal_graph_list:
        oi = entity_map[graph["owner_entity_id"]]
        owner_entity.append(oi)
        owner_root.append(root_occurrence[oi])
        owned_graph[oi] = graph_map[graph["graph_id"]]
    has_owner = [True] * len(internal_graph_list)
    has_owner_root = [idx >= 0 for idx in owner_root]

    edge_lists: dict[str, list[RoutedFact]] = {scope: [] for scope in SCOPES}
    counts_by_split = {split: Counter() for split in sorted(SPLITS)}
    for fact in facts:
        scope = scope_by_fact[fact["fact_id"]]
        counts_by_split[fact["split"]][scope] += 1
        if fact["split"] != "train":
            continue
        source = node_by_id[fact["source_node_id"]]
        target = node_by_id[fact["target_node_id"]]
        edge_lists[scope].append(RoutedFact(
            fact_id=fact["fact_id"], relation_type_id=fact["relation_type_id"],
            relation_index=relation_map[fact["relation_type_id"]],
            source_node_id=fact["source_node_id"], target_node_id=fact["target_node_id"],
            source_occurrence_index=node_map[fact["source_node_id"]],
            target_occurrence_index=node_map[fact["target_node_id"]],
            source_graph_id=source["graph_id"], target_graph_id=target["graph_id"],
            scope=scope, split=fact["split"], direction="FORWARD"))

    report = {
        "validation": "passed",
        "dataset_name": data.get("fixture_name", data.get("dataset_name", "unnamed_dataset")),
        "counts": {"entities": len(entities), "entity_types": len({e.get("type_id") for e in entities}),
                   "graphs": len(graphs), "internal_graphs": len(internal_graph_list),
                   "graph_nodes": len(nodes), "facts": len(facts)},
        "scope_counts_by_split": {split: {scope: counts_by_split[split][scope] for scope in SCOPES} for split in sorted(SPLITS)},
        "training_edge_counts_by_scope": {scope: len(edge_lists[scope]) for scope in SCOPES},
        "held_out_fact_ids": [f["fact_id"] for f in facts if f["split"] != "train"],
        "reverse_channels_generated": False,
        "splits_generated": data.get("splits_generated", False),
        "split_seed": data.get("split_seed"),
        "split_unit": "unordered canonical entity pair" if data.get("splits_generated", False) else "source-provided",
        "candidate_type_rules": "provided relation schema" if schema else "none; graph context only",
    }
    type_indices = [type_map.get(entity.get("type_id"), type_map["UNK"]) for entity in entities]
    report["type_ids"] = type_ids
    return PreprocessedGoG(entities, graphs, nodes, facts, schema, entity_map, graph_map, node_map, relation_map,
                           type_map, type_indices,
                           occurrence_entity, occurrence_graph, root_occurrence, has_root, owner_entity,
                           has_owner, owner_root, has_owner_root, owned_graph,
                           [idx >= 0 for idx in owned_graph], edge_lists, report)


def load_gog(path: str | Path, split_seed: int = 42) -> PreprocessedGoG:
    """Load JSON fixture and preprocess it."""
    with Path(path).open(encoding="utf-8") as stream:
        return preprocess_data(json.load(stream), split_seed=split_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preprocess a GoG JSON dataset")
    parser.add_argument("fixture", type=Path, help="Input GoG JSON file")
    parser.add_argument("--output", type=Path, help="Optional path for complete preprocessed JSON output")
    args = parser.parse_args()
    result = load_gog(args.fixture)
    print(json.dumps(result.report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
