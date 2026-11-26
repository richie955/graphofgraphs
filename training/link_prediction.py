"""Small, reusable Link prediction training and ranking evaluation helpers."""

from __future__ import annotations

import json
import hashlib
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gog_embedding import ComplExDecoder, GoGEncoder
from baselines.rgcn import RGCNBaseline
from gog_embedding.stage1 import Stage1Initializer
from data_pipeline.preprocessing import PreprocessedGoG
from .checkpoints import initialize_from_checkpoint
from gog_embedding.structural import ELEMENTS, FRAGMENTS


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 256
    num_bases: int = 8
    num_heads: int = 4
    dropout: float = 0.2
    num_rounds: int | None = None
    negatives_per_side: int = 10
    training_batch_size: int = 4096
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 17
    device: str = "auto"
    show_progress: bool = False
    encoder_profile: str = "legacy"
    task_scope: str | None = None
    feature_mode: str = "auto"
    basis_init: str | None = None
    composition: str | None = None
    normalization: str | None = None
    identity_residual: bool = True
    local_enabled: bool = True
    cross_enabled: bool = True
    fusion_enabled: bool = True
    root_enabled: bool = True
    sync_enabled: bool = False
    feedback_enabled: bool = False
    family_readout: bool = True
    embedding_learning_rate: float | None = None
    encoder_learning_rate: float | None = None
    decoder_learning_rate: float | None = None
    init_checkpoint: str | None = None
    early_stopping: bool = True
    update_offset: int = 0
    evaluate_test: bool | None = None
    diagnostic_train_limit: int = 32

    def __post_init__(self) -> None:
        if self.encoder_profile not in {"legacy", "structural"}:
            raise ValueError("encoder_profile must be legacy or structural")
        if self.task_scope is None:
            object.__setattr__(self, "task_scope", "root" if self.encoder_profile == "structural" else "all")
        if self.evaluate_test is None:
            object.__setattr__(self, "evaluate_test", self.encoder_profile == "legacy")
        if self.task_scope not in {"all", "root"}:
            raise ValueError("task_scope must be all or root")


def selected_facts(data: PreprocessedGoG, split: str, task_scope: str = "all") -> list[dict[str, Any]]:
    if task_scope not in {"all", "root"}:
        raise ValueError("task_scope must be all or root")
    root = next(g["graph_id"] for g in data.graphs if g["kind"] == "root")
    nodes = data.graph_node_by_id
    return [f for f in data.facts if f["split"] == split and not f.get("derived_channel", False)
            and f.get("supervised", True) and (task_scope == "all" or
                (nodes[f["source_node_id"]]["graph_id"] == root and
                 nodes[f["target_node_id"]]["graph_id"] == root))]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def protocol_fingerprints(data: PreprocessedGoG, task_scope: str) -> dict[str, Any]:
    root = next(g["graph_id"] for g in data.graphs if g["kind"] == "root")
    def triples(split: str) -> list:
        return sorted((f["source_node_id"], f["relation_type_id"], f["target_node_id"])
                      for f in selected_facts(data, split, task_scope))
    return {
        "dataset": _fingerprint({"entities": sorted(data.entities, key=lambda r: r["entity_id"]),
                                  "graphs": sorted(data.graphs, key=lambda r: r["graph_id"]),
                                  "nodes": sorted(data.graph_nodes, key=lambda r: r["node_id"]),
                                  "facts": sorted(data.facts, key=lambda r: r["fact_id"]),
                                  "relation_schema": data.relation_schema}),
        "candidates": _fingerprint({"nodes": sorted(n["node_id"] for n in data.graph_nodes
                                                      if task_scope == "all" or n["graph_id"] == root),
                                     "schema": data.relation_schema if task_scope == "all" else {},
                                     "task_scope": task_scope}),
        "queries": {s: _fingerprint(triples(s)) for s in ("train", "validation", "test")},
    }


class EntityOnlyBaseline(nn.Module):
    """Stage 1 vectors only; the decoder is shared with the GoG experiment."""

    def __init__(self, num_entities: int, num_types: int, hidden_dim: int):
        super().__init__()
        self.initializer = Stage1Initializer(num_entities, num_types, hidden_dim)

    def forward(self, data: PreprocessedGoG) -> tuple[Tensor, Tensor]:
        device = self.initializer.entity_embedding.weight.device
        types = torch.tensor(data.type_index_by_entity, dtype=torch.long, device=device)
        occ_entities = torch.tensor(data.entity_index_by_occurrence, dtype=torch.long, device=device)
        b_entity, _, z_occ = self.initializer(types, occ_entities)
        return b_entity, z_occ


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _indexes(data: PreprocessedGoG) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[int, list[int]]]:
    node_by_id = data.graph_node_by_id
    entity_by_id = {row["entity_id"]: row for row in data.entities}
    nodes_by_graph: dict[int, list[int]] = defaultdict(list)
    for node_id, node_idx in data.node_id_to_index.items():
        nodes_by_graph[data.graph_id_to_index[node_by_id[node_id]["graph_id"]]].append(node_idx)
    return data.node_id_to_index, entity_by_id, nodes_by_graph


def _fact_context(fact: dict[str, Any], node_by_id: dict[str, dict[str, Any]]) -> tuple[str, str]:
    return node_by_id[fact["source_node_id"]]["graph_id"], node_by_id[fact["target_node_id"]]["graph_id"]


def known_fact_keys(data: PreprocessedGoG, splits: set[str]) -> set[tuple[str, str, str, str, str]]:
    node_by_id = data.graph_node_by_id
    result = set()
    for fact in data.facts:
        if fact["split"] in splits:
            source_graph, target_graph = _fact_context(fact, node_by_id)
            result.add((fact["source_node_id"], fact["relation_type_id"], fact["target_node_id"], source_graph, target_graph))
    return result


def valid_candidate_nodes(data: PreprocessedGoG, fact: dict[str, Any], side: str,
                          split_known_keys: set[tuple[str, str, str, str, str]]) -> list[int]:
    """Return type- and graph-valid corruption nodes, excluding known truths."""
    node_by_id = data.graph_node_by_id
    source_graph, target_graph = _fact_context(fact, node_by_id)
    # A source KG may not provide endpoint-type rules. In that case the
    # candidate universe is limited by the exact graph context only.
    relation = data.relation_schema.get(fact["relation_type_id"], {})
    required_graph = source_graph if side == "head" else target_graph
    allowed_types = relation.get("head_types" if side == "head" else "tail_types")
    positive_endpoint = fact["source_node_id"] if side == "head" else fact["target_node_id"]
    if allowed_types:
        candidate_pool = sorted({index for entity_type in allowed_types
                                 for index in data.candidate_node_indices_by_graph_type.get(
                                     (required_graph, entity_type), [])})
    else:
        candidate_pool = data.candidate_node_indices_by_graph.get(required_graph, [])
    candidates: list[int] = []
    for index in candidate_pool:
        node = data.graph_nodes[index]
        candidate_node = node["node_id"]
        if side == "head":
            key = (candidate_node, fact["relation_type_id"], fact["target_node_id"], source_graph, target_graph)
        else:
            key = (fact["source_node_id"], fact["relation_type_id"], candidate_node, source_graph, target_graph)
        if candidate_node == positive_endpoint or key in split_known_keys:
            continue
        candidates.append(data.node_id_to_index[candidate_node])
    return sorted(candidates)


def _sample_side_candidates(data: PreprocessedGoG, fact: dict[str, Any], side: str,
                            negatives_per_side: int, rng: random.Random,
                            known_by_head: dict[tuple[str, str, str, str], set[str]],
                            known_by_tail: dict[tuple[str, str, str, str], set[str]],
                            all_candidates: bool = False) -> list[int]:
    """Sample negatives by graph index and rejection sampling, avoiding a full
    candidate scan for every training fact (especially useful for ROOT facts)."""
    if negatives_per_side <= 0:
        return []
    node_by_id = data.graph_node_by_id
    source_graph, target_graph = _fact_context(fact, node_by_id)
    required_graph = source_graph if side == "head" else target_graph
    relation = data.relation_schema.get(fact["relation_type_id"], {})
    allowed_types = None if all_candidates else relation.get("head_types" if side == "head" else "tail_types")
    if allowed_types:
        pool = sorted({index for entity_type in allowed_types
                       for index in data.candidate_node_indices_by_graph_type.get((required_graph, entity_type), [])})
    else:
        pool = data.candidate_node_indices_by_graph.get(required_graph, [])
    positive_id = fact["source_node_id"] if side == "head" else fact["target_node_id"]
    key = ((fact["relation_type_id"], fact["target_node_id"], source_graph, target_graph)
           if side == "head" else
           (fact["relation_type_id"], fact["source_node_id"], source_graph, target_graph))
    blocked = known_by_head.get(key, set()) if side == "head" else known_by_tail.get(key, set())
    if len(pool) <= 512:
        valid = [index for index in pool
                 if data.graph_nodes[index]["node_id"] != positive_id
                 and data.graph_nodes[index]["node_id"] not in blocked]
        return rng.sample(valid, min(negatives_per_side, len(valid)))

    selected: set[int] = set()
    attempts = max(128, negatives_per_side * 32)
    for _ in range(attempts):
        index = pool[rng.randrange(len(pool))]
        node_id = data.graph_nodes[index]["node_id"]
        if node_id != positive_id and node_id not in blocked:
            selected.add(index)
            if len(selected) == negatives_per_side:
                break
    if len(selected) < negatives_per_side:
        # Fallback only when the context is unusually dense or tiny after
        # filtering; the usual ROOT path avoids scanning the whole node list.
        valid = [index for index in pool
                 if data.graph_nodes[index]["node_id"] != positive_id
                 and data.graph_nodes[index]["node_id"] not in blocked
                 and index not in selected]
        remaining = min(negatives_per_side - len(selected), len(valid))
        selected.update(rng.sample(valid, remaining))
    return sorted(selected)


def sample_training_negatives(data: PreprocessedGoG, negatives_per_side: int, seed: int,
                              task_scope: str = "all") -> tuple[list[tuple[int, str, int]], dict[str, int]]:
    """Return (positive fact index, side, replacement node index) triples."""
    rng = random.Random(seed)
    train_facts = selected_facts(data, "train", task_scope)
    train_known = known_fact_keys(data, {"train"})
    known_by_head: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    known_by_tail: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for head, relation, tail, source_graph, target_graph in train_known:
        known_by_head[(relation, tail, source_graph, target_graph)].add(head)
        known_by_tail[(relation, head, source_graph, target_graph)].add(tail)
    sampled: list[tuple[int, str, int]] = []
    shortfalls = {"head": 0, "tail": 0, "no_negative_side": 0}
    for fact_index, fact in enumerate(train_facts):
        found_side = False
        for side in ("head", "tail"):
            candidates = _sample_side_candidates(data, fact, side, negatives_per_side, rng,
                                                 known_by_head, known_by_tail, all_candidates=task_scope == "root")
            if candidates:
                found_side = True
                for candidate in candidates:
                    sampled.append((fact_index, side, candidate))
            else:
                shortfalls[side] += 1
        if not found_side:
            shortfalls["no_negative_side"] += 1
    return sampled, shortfalls


def _training_loss(data: PreprocessedGoG, decoder: ComplExDecoder, occurrence_states: Tensor,
                   negatives: list[tuple[int, str, int]], train_facts: list[dict[str, Any]],
                   node_map: dict[str, int], relation_map: dict[str, int]) -> Tensor:
    device = occurrence_states.device
    heads = torch.tensor([node_map[f["source_node_id"]] for f in train_facts], dtype=torch.long, device=device)
    tails = torch.tensor([node_map[f["target_node_id"]] for f in train_facts], dtype=torch.long, device=device)
    rels = torch.tensor([relation_map[f["relation_type_id"]] for f in train_facts], dtype=torch.long, device=device)
    positive_scores = decoder.score(occurrence_states[heads], rels, occurrence_states[tails])
    pos_loss = F.softplus(-positive_scores)
    if not negatives:
        return pos_loss.mean()
    owner_rows = torch.tensor([n[0] for n in negatives], dtype=torch.long, device=device)
    neg_heads = torch.tensor([n[2] if n[1] == "head" else node_map[train_facts[n[0]]["source_node_id"]]
                              for n in negatives], dtype=torch.long, device=device)
    neg_tails = torch.tensor([n[2] if n[1] == "tail" else node_map[train_facts[n[0]]["target_node_id"]]
                              for n in negatives], dtype=torch.long, device=device)
    neg_rels = rels[owner_rows]
    neg_scores = decoder.score(occurrence_states[neg_heads], neg_rels, occurrence_states[neg_tails])
    neg_loss_sum = pos_loss.new_zeros(len(train_facts))
    neg_counts = pos_loss.new_zeros(len(train_facts))
    neg_loss_sum.index_add_(0, owner_rows, F.softplus(neg_scores))
    neg_counts.index_add_(0, owner_rows, torch.ones_like(neg_scores))
    has_neg = neg_counts > 0
    per_fact = pos_loss + torch.where(has_neg, neg_loss_sum / neg_counts.clamp_min(1), torch.zeros_like(pos_loss))
    return per_fact.mean()


def _backward_in_fact_batches(data: PreprocessedGoG, decoder: ComplExDecoder,
                              occurrence_states: Tensor,
                              negatives: list[tuple[int, str, int]],
                              train_facts: list[dict[str, Any]], node_map: dict[str, int],
                              relation_map: dict[str, int], batch_size: int) -> float:
    """Accumulate decoder batches, then backpropagate into the encoder once.

    This avoids keeping activations for every negative triple on the GPU while
    also avoiding a repeated backward pass through the full GoG encoder.
    """
    if batch_size < 1:
        raise ValueError("training_batch_size must be positive")
    leaf_states = occurrence_states.detach().requires_grad_(True)
    negatives_by_fact: list[list[tuple[str, int]]] = [[] for _ in train_facts]
    for fact_index, side, candidate_index in negatives:
        negatives_by_fact[fact_index].append((side, candidate_index))

    weighted_loss = 0.0
    fact_count = len(train_facts)
    for start in range(0, fact_count, batch_size):
        end = min(start + batch_size, fact_count)
        batch_facts = train_facts[start:end]
        batch_negatives = [
            (fact_index - start, side, candidate_index)
            for fact_index in range(start, end)
            for side, candidate_index in negatives_by_fact[fact_index]
        ]
        loss = _training_loss(data, decoder, leaf_states, batch_negatives,
                              batch_facts, node_map, relation_map)
        weight = (end - start) / fact_count
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight

    if leaf_states.grad is not None:
        occurrence_states.backward(leaf_states.grad)
    return weighted_loss


def _scope_group(scope: str) -> str:
    if scope == "root_to_root":
        return "root"
    if scope == "same_internal_graph":
        return "internal"
    return "cross_context"


@torch.no_grad()
def evaluate(data: PreprocessedGoG, decoder: ComplExDecoder, occurrence_states: Tensor,
             split: str, known_splits: set[str], task_scope: str = "all",
             fact_limit: int | None = None) -> dict[str, Any]:
    """Rank candidates in query batches, projecting each node only once."""
    device = occurrence_states.device
    node_map = data.node_id_to_index
    node_by_id = data.graph_node_by_id
    known = known_fact_keys(data, known_splits)
    ranks: dict[str, dict[str, list[float]]] = {
        group: {"raw": [], "filtered": []} for group in ("root", "internal", "cross_context")
    }
    unscorable = {group: 0 for group in ranks}
    total = {group: 0 for group in ranks}
    breakdowns = {key: defaultdict(lambda: {"raw": [], "filtered": [], "total_queries": 0})
                  for key in ("by_relation", "by_side", "by_owner")}
    owners = {g["owner_entity_id"] for g in data.graphs if g["kind"] == "internal"}

    def categories(fact: dict, side: str) -> dict[str, str]:
        touches = any(node_by_id[fact[k]]["entity_id"] in owners
                      for k in ("source_node_id", "target_node_id"))
        return {"by_relation": fact["relation_type_id"], "by_side": side,
                "by_owner": "owner_touching" if touches else "non_owner"}
    known_heads: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    known_tails: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for head, relation, tail, source_graph, target_graph in known:
        known_heads[(relation, tail, source_graph, target_graph)].add(head)
        known_tails[(relation, head, source_graph, target_graph)].add(tail)

    # Queries share candidate pools when relation and endpoint contexts match.
    query_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    root_graph_id = next(g["graph_id"] for g in data.graphs if g["kind"] == "root")
    facts = selected_facts(data, split, task_scope)
    if fact_limit is not None:
        facts = sorted(facts, key=lambda f: f["fact_id"])[:fact_limit]
    for fact in facts:
        source_graph, target_graph = _fact_context(fact, node_by_id)
        group = ("root" if source_graph == target_graph == root_graph_id else
                 "internal" if source_graph == target_graph else "cross_context")
        head_index = node_map[fact["source_node_id"]]
        tail_index = node_map[fact["target_node_id"]]
        relation_index = data.relation_id_to_index[fact["relation_type_id"]]
        for side in ("head", "tail"):
            required_graph = source_graph if side == "head" else target_graph
            relation = data.relation_schema.get(fact["relation_type_id"], {})
            allowed_types = (() if task_scope == "root" else
                             tuple(relation.get("head_types" if side == "head" else "tail_types", [])))
            key = (side, relation_index, source_graph, target_graph, required_graph, allowed_types)
            query_groups[key].append({
                "fact": fact, "group": group,
                "fixed_index": tail_index if side == "head" else head_index,
                "true_index": head_index if side == "head" else tail_index,
            })
            total[group] += 1
            for category, label in categories(fact, side).items():
                breakdowns[category][label]["total_queries"] += 1

    projected = decoder.project_for_ranking(occurrence_states)
    query_batch_size = 64
    for key, rows in query_groups.items():
        side, relation_index, source_graph, target_graph, required_graph, allowed_types = key
        if allowed_types:
            candidates = sorted({index for entity_type in allowed_types
                                 for index in data.candidate_node_indices_by_graph_type.get(
                                     (required_graph, entity_type), [])})
        else:
            candidates = data.candidate_node_indices_by_graph.get(required_graph, [])
        candidate_position = {index: column for column, index in enumerate(candidates)}
        scorable_rows = []
        for row in rows:
            if len(candidates) < 2 or row["true_index"] not in candidate_position:
                unscorable[row["group"]] += 1
            else:
                scorable_rows.append(row)
        if not scorable_rows:
            continue
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        relation_ids = torch.full((min(query_batch_size, len(scorable_rows)),),
                                  relation_index, dtype=torch.long, device=device)
        for start in range(0, len(scorable_rows), query_batch_size):
            batch = scorable_rows[start:start + query_batch_size]
            count = len(batch)
            fixed = torch.tensor([row["fixed_index"] for row in batch], dtype=torch.long, device=device)
            batch_relations = relation_ids[:count]
            scores = decoder.score_candidate_matrix(projected, fixed, batch_relations,
                                                    candidate_tensor, side)
            true_columns = torch.tensor([candidate_position[row["true_index"]] for row in batch],
                                        dtype=torch.long, device=device)
            row_ids = torch.arange(count, device=device)
            positive_scores = scores[row_ids, true_columns]
            above = scores > positive_scores[:, None]
            raw_ranks = 1 + above.sum(dim=1)

            # Filter known true alternatives via the sparse known-edge index.
            blocked_rows: list[int] = []
            blocked_columns: list[int] = []
            for row_id, row in enumerate(batch):
                fact = row["fact"]
                blocked_ids = (known_heads.get((fact["relation_type_id"], fact["target_node_id"],
                                                source_graph, target_graph), set()) if side == "head" else
                               known_tails.get((fact["relation_type_id"], fact["source_node_id"],
                                                source_graph, target_graph), set()))
                for blocked_id in blocked_ids:
                    blocked_index = node_map.get(blocked_id)
                    column = candidate_position.get(blocked_index) if blocked_index is not None else None
                    if column is not None:
                        blocked_rows.append(row_id)
                        blocked_columns.append(column)
            known_mask = torch.zeros_like(above)
            if blocked_rows:
                known_mask[torch.tensor(blocked_rows, device=device),
                           torch.tensor(blocked_columns, device=device)] = True
            filtered_ranks = raw_ranks - (above & known_mask).sum(dim=1)
            raw_values = raw_ranks.cpu().tolist()
            filtered_values = filtered_ranks.cpu().tolist()
            for row, raw_rank, filtered_rank in zip(batch, raw_values, filtered_values):
                ranks[row["group"]]["raw"].append(float(raw_rank))
                ranks[row["group"]]["filtered"].append(float(filtered_rank))
                for category, label in categories(row["fact"], side).items():
                    breakdowns[category][label]["raw"].append(float(raw_rank))
                    breakdowns[category][label]["filtered"].append(float(filtered_rank))

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mrr": None, "mean_rank": None, "hits@1": None, "hits@3": None, "hits@10": None}
        tensor = torch.tensor(values, dtype=torch.float64)
        return {"count": len(values), "mrr": float((1 / tensor).mean()), "mean_rank": float(tensor.mean()),
                "hits@1": float((tensor <= 1).double().mean()), "hits@3": float((tensor <= 3).double().mean()),
                "hits@10": float((tensor <= 10).double().mean())}

    result = {group: {kind: summarize(values) for kind, values in kinds.items()} for group, kinds in ranks.items()}
    for group in result:
        result[group]["total_queries"] = total[group]
        result[group]["unscorable_queries"] = unscorable[group]
        result[group]["scorable_queries"] = len(ranks[group]["raw"])
    for kind in ("raw", "filtered"):
        available = [result[g][kind]["mrr"] for g in ranks if result[g][kind]["mrr"] is not None]
        result.setdefault("aggregate", {})[kind] = {
            "micro": summarize([rank for group in ranks.values() for rank in group[kind]])["mrr"],
            "macro_scope_mrr": sum(available) / len(available) if available else None,
        }
    result["settings"] = {"split": split, "filter_known_splits": sorted(known_splits),
                          "rank_tie_rule": "optimistic (strictly greater scores rank above target)",
                          "task_scope": task_scope, "fact_limit": fact_limit,
                          "root_candidate_count": (len(data.candidate_node_indices_by_graph.get(root_graph_id, []))
                                                   if task_scope == "root" else None),
                          "candidates": "all ROOT occurrences" if task_scope == "root" else "graph/type-valid"}
    result["breakdowns"] = {category: {label: {"raw": summarize(values["raw"]),
                                                "filtered": summarize(values["filtered"]),
                                                "total_queries": values["total_queries"],
                                                "unscorable_queries": values["total_queries"] - len(values["raw"])}
                                        for label, values in rows.items()}
                             for category, rows in breakdowns.items()}
    return result


def _gradient_norms(encoder: nn.Module, decoder: nn.Module) -> dict[str, float]:
    groups: dict[str, list[Tensor]] = defaultdict(list)
    for name, parameter in encoder.named_parameters():
        if parameter.grad is not None:
            groups[name.split(".")[0]].append(parameter.grad.detach().norm())
    groups["decoder"] = [p.grad.detach().norm() for p in decoder.parameters() if p.grad is not None]
    return {key: float(torch.stack(values).norm()) if values else 0.0 for key, values in groups.items()}


def _encoder_diagnostics(encoder: nn.Module, data: PreprocessedGoG, encoded: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("local", "root"):
        module = getattr(encoder, name, None)
        if module is not None:
            result[name + "_message_self_ratio"] = module.last_message_self_ratio
    scale = getattr(encoder, "identity_scale", None)
    if scale is not None:
        result["identity_scales"] = {key: float(scale[index].detach())
                                     for key, index in data.type_id_to_index.items()}
    if hasattr(encoder, "feedback_scale"):
        result["feedback_scale"] = float(encoder.feedback_scale.detach())
    if encoded is not None:
        result["round_changes"] = encoder.last_round_changes
    return result


def _run_one(data: PreprocessedGoG, config: TrainingConfig, kind: str, output_dir: Path) -> dict[str, Any]:
    if config.epochs < 1 or config.update_offset < 0 or config.diagnostic_train_limit < 0:
        raise ValueError("epochs must be positive; update offset and diagnostic limit must be nonnegative")
    if config.early_stopping and config.patience < 1:
        raise ValueError("patience must be positive when early stopping is enabled")
    if config.gradient_clip <= 0 or config.weight_decay < 0:
        raise ValueError("gradient clipping must be positive and weight decay nonnegative")
    started = time.perf_counter()
    seed = config.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(config.device)
    if kind == "gog":
        encoder: nn.Module = GoGEncoder.from_preprocessed(
            data, hidden_dim=config.hidden_dim, num_bases=config.num_bases,
            num_heads=config.num_heads, dropout=config.dropout, num_rounds=config.num_rounds,
            profile=config.encoder_profile, feature_mode=config.feature_mode,
            basis_init=config.basis_init, composition=config.composition,
            normalization=config.normalization, identity_residual=config.identity_residual,
            local_enabled=config.local_enabled, cross_enabled=config.cross_enabled,
            fusion_enabled=config.fusion_enabled, root_enabled=config.root_enabled,
            sync_enabled=config.sync_enabled, feedback_enabled=config.feedback_enabled,
            family_readout=config.family_readout)
    elif kind == "baseline":
        encoder = EntityOnlyBaseline(len(data.entities), len(data.type_id_to_index), config.hidden_dim)
    elif kind == "rgcn":
        encoder = RGCNBaseline.from_preprocessed(
            data, hidden_dim=config.hidden_dim, num_bases=config.num_bases,
            dropout=config.dropout)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    decoder = ComplExDecoder(len(data.relation_id_to_index), config.hidden_dim)
    encoder.to(device)
    decoder.to(device)
    initialization = ({"mode": "random"} if config.init_checkpoint is None else
                      initialize_from_checkpoint(encoder, decoder, data, config.init_checkpoint))
    embedding_parameters = list(encoder.initializer.parameters())
    embedding_ids = {id(p) for p in embedding_parameters}
    encoder_parameters = [p for p in encoder.parameters() if id(p) not in embedding_ids]
    optimizer_groups = [{"params": embedding_parameters, "lr": config.embedding_learning_rate
                         if config.embedding_learning_rate is not None else config.learning_rate},
                        {"params": encoder_parameters, "lr": config.encoder_learning_rate
                         if config.encoder_learning_rate is not None else config.learning_rate},
                        {"params": list(decoder.parameters()), "lr": config.decoder_learning_rate
                         if config.decoder_learning_rate is not None else config.learning_rate}]
    if any(group["lr"] <= 0 for group in optimizer_groups):
        raise ValueError("learning rates must be positive")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config.weight_decay)
    node_map = data.node_id_to_index
    train_facts = selected_facts(data, "train", config.task_scope)
    if not train_facts:
        raise ValueError("training split contains no facts")
    history: list[float] = []
    best_mrr = float("-inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    diagnostics: list[dict[str, Any]] = []
    negative_fingerprints: list[str] = []
    fingerprints = protocol_fingerprints(data, config.task_scope)

    def encode() -> tuple[Tensor, Any]:
        if kind == "gog":
            output = encoder(data)
            return output.z_occurrence, output
        return encoder(data)[1], None

    def snapshot() -> dict[str, Any]:
        return {"encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}}

    initial_validation = None
    if config.init_checkpoint is not None:
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            states, _ = encode()
            initial_validation = evaluate(data, decoder, states, "validation", {"train", "validation"},
                                           config.task_scope)
        score = initial_validation["aggregate"]["filtered"]["macro_scope_mrr"]
        if score is not None:
            best_mrr, best_state = float(score), snapshot()
    shortfalls = {"head": 0, "tail": 0, "no_negative_side": 0}
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        if config.show_progress:
            print(f"[{kind}] epoch {epoch + 1}/{config.epochs}: sampling negatives and training...", flush=True)
        negatives, shortfalls = sample_training_negatives(data, config.negatives_per_side,
                                                          seed + (config.update_offset + epoch) * 1009,
                                                          config.task_scope)
        negative_fingerprints.append(_fingerprint([(train_facts[i]["fact_id"], side,
                                                    data.graph_nodes[node]["node_id"])
                                                   for i, side, node in negatives]))
        encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        occurrence_states, encoded = encode()
        loss_value = _backward_in_fact_batches(
            data, decoder, occurrence_states, negatives, train_facts,
            node_map, data.relation_id_to_index, config.training_batch_size)
        if not torch.isfinite(torch.tensor(loss_value)):
            raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
        step_diagnostics = _encoder_diagnostics(encoder, data, encoded)
        step_diagnostics["gradient_norms_before_clip"] = _gradient_norms(encoder, decoder)
        clipped = nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], config.gradient_clip,
                                          error_if_nonfinite=True)
        step_diagnostics["total_gradient_norm_before_clip"] = float(clipped)
        optimizer.step()
        history.append(loss_value)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_states, _ = encode()
            val_metrics = evaluate(data, decoder, val_states, "validation", {"train", "validation"},
                                   config.task_scope)
            if config.diagnostic_train_limit:
                train_metrics = evaluate(data, decoder, val_states, "train", {"train"},
                                         config.task_scope, fact_limit=config.diagnostic_train_limit)
                step_diagnostics["training_ranking"] = train_metrics["aggregate"]
        score = val_metrics["aggregate"]["filtered"]["macro_scope_mrr"]
        score = float(score) if score is not None else -float("inf")
        step_diagnostics.update(update=config.update_offset + epoch + 1, loss=loss_value,
                                validation_filtered_mrr=score if score != -float("inf") else None)
        diagnostics.append(step_diagnostics)
        if score > best_mrr:
            best_mrr, best_epoch, stale = score, epoch + 1, 0
            best_state = snapshot()
        else:
            stale += 1
        if config.show_progress:
            elapsed = time.perf_counter() - epoch_start
            print(f"[{kind}] epoch {epoch + 1}/{config.epochs} "
                  f"loss={history[-1]:.4f} validation_filtered_MRR={score:.4f} "
                  f"best_epoch={best_epoch} elapsed={elapsed:.1f}s", flush=True)
        if config.early_stopping and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a selectable validation checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        final_states, _ = encode()
        validation = evaluate(data, decoder, final_states, "validation", {"train", "validation"}, config.task_scope)
        test = (evaluate(data, decoder, final_states, "test", {"train", "validation", "test"}, config.task_scope)
                if config.evaluate_test else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{kind}_best.pt"
    torch.save({"encoder": best_state["encoder"], "decoder": best_state["decoder"],
                "config": asdict(config), "kind": kind, "best_epoch": best_epoch,
                "type_id_to_index": data.type_id_to_index, "fingerprints": fingerprints,
                "feature_vocabularies": ({"elements": ELEMENTS, "fragments": FRAGMENTS}
                                          if getattr(encoder, "chemistry", None) is not None else None),
                "initialization": initialization, "optimizer_reset_on_initialization": True,
                "entity_id_to_index": data.entity_id_to_index,
                "node_id_to_index": data.node_id_to_index,
                "relation_id_to_index": data.relation_id_to_index,
                "graph_id_to_index": data.graph_id_to_index}, checkpoint_path)
    return {"kind": kind, "best_epoch": best_epoch, "best_validation_macro_filtered_mrr": best_mrr,
            "loss_first": history[0], "loss_last": history[-1], "loss_history": history,
            "negative_shortfalls_last_epoch": shortfalls,
            "validation": validation, "test": test, "checkpoint": str(checkpoint_path),
            "optimizer_updates": len(history), "update_offset": config.update_offset,
            "initial_validation": initial_validation, "initialization": initialization,
            "fingerprints": fingerprints, "negative_sample_fingerprints": negative_fingerprints,
            "diagnostics": diagnostics, "effective_rounds": getattr(encoder, "num_rounds", None),
            "training_target_count": len(train_facts), "elapsed_seconds": time.perf_counter() - started,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__)}}


def run_baseline(data: PreprocessedGoG, config: TrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train only the entity-and-type baseline and save its validation/test report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    baseline = _run_one(data, config, "baseline", output)
    result = {
        "kind": "entity_only_baseline",
        "notice": "Initial result on the provided split; not a benchmark claim.",
        "dataset_name": data.report.get("dataset_name", "dataset"),
        "split_note": "The PrimeKG GoG split is generated and should be treated as an initial pipeline check.",
        "config": asdict(config),
        "split_counts": data.report.get("scope_counts_by_split", {}),
        "baseline": baseline,
        "elapsed_seconds": time.time() - start,
    }
    (output / "baseline_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_gog(data: PreprocessedGoG, config: TrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train only GoG, allowing it to be compared with a previously saved baseline run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    gog = _run_one(data, config, "gog", output)
    result = {
        "kind": "gog",
        "notice": "Initial result on the provided split; compare against a baseline with matching settings.",
        "dataset_name": data.report.get("dataset_name", "dataset"),
        "split_note": "The PrimeKG GoG split is generated and should be treated as an initial pipeline check.",
        "config": asdict(config),
        "split_counts": data.report.get("scope_counts_by_split", {}),
        "gog": gog,
        "elapsed_seconds": time.time() - start,
    }
    (output / "gog_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_rgcn(data: PreprocessedGoG, config: TrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train the flat R-GCN using the same decoder and evaluation protocol."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rgcn = _run_one(data, config, "rgcn", output)
    result = {
        "kind": "rgcn",
        "notice": "Exploratory result on the provided split; not a benchmark claim.",
        "dataset_name": data.report.get("dataset_name", "dataset"),
        "config": asdict(config),
        "split_counts": data.report.get("scope_counts_by_split", {}),
        "rgcn": rgcn,
        "elapsed_seconds": time.time() - start,
    }
    (output / "rgcn_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_comparison(data: PreprocessedGoG, config: TrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train baseline and full GoG with the same splits and candidate rules."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    baseline = _run_one(data, config, "baseline", output)
    gog = _run_one(data, config, "gog", output)
    dataset_name = data.report.get("dataset_name", "dataset")
    notice = ("Exploratory results on an automatically generated split; not a benchmark claim."
              if data.report.get("splits_generated") else
              "Synthetic fixture results check software behavior only; they are not evidence of real-world improvement."
              if "fixture" in dataset_name.lower() or "synthetic" in dataset_name.lower() else
              "Results use the dataset's supplied split; review its evaluation protocol before making a benchmark claim.")
    result = {
        "notice": notice,
        "dataset_name": dataset_name,
        "split_note": "Use benchmark-prescribed splits for real evaluation.",
        "config": asdict(config),
        "training_facts": [f["fact_id"] for f in selected_facts(data, "train", config.task_scope)],
        "derived_training_channels": sum(f["split"] == "train" and f.get("derived_channel", False)
                                         for f in data.facts),
        "validation_facts": [f["fact_id"] for f in selected_facts(data, "validation", config.task_scope)],
        "test_facts": [f["fact_id"] for f in selected_facts(data, "test", config.task_scope)],
        "baseline": baseline,
        "gog": gog,
        "elapsed_seconds": time.time() - start,
    }
    (output / "comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
