"""Fixed-source-split link prediction pilot for the existing PrimeKG GoG."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from baselines.rgcn import RGCNBaseline
from gog_data.preprocessing import PreprocessedGoG, preprocess_data
from gog_model import ComplExDecoder, GoGEncoder
from gog_model.stage1 import Stage1Initializer
from training.link_prediction import TrainingConfig, _backward_in_fact_batches, choose_device


EXPECTED_SOURCE = {"train": 160_000, "validation": 20_000, "test": 20_000}
EXPECTED_COVERED = {"train": 2_936, "validation": 226, "test": 256}
MODEL_KINDS = ("flat_root", "flat_context", "gog")


@dataclass(frozen=True)
class PilotFact:
    fact_id: str
    head: int
    relation: int
    tail: int
    relation_label: str
    pair: tuple[str, str]


@dataclass
class CoveredExperiment:
    gog: PreprocessedGoG
    flat: PreprocessedGoG
    facts: dict[str, list[PilotFact]]
    entity_ids: list[str]
    relation_labels: list[str]
    typed_candidates: dict[tuple[int, str], tuple[int, ...]]
    hashes: dict[str, str]
    audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(row: dict[str, Any]) -> str:
    broad = str(row["relation"])
    display = row.get("display_relation")
    return f"{broad}::{display}" if display else broad


def _pair(head: str, tail: str) -> tuple[str, str]:
    return tuple(sorted((str(head), str(tail))))


def _source_key(head: str, tail: str, relation: str, display: str) -> tuple[Any, ...]:
    return (_pair(head, tail), str(relation), str(display))


def _read_split(path: Path, split: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["facts"]
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"{path}: incorrect split label")
    if data.get("metadata", {}).get("total_facts") != len(rows):
        raise ValueError(f"{path}: metadata count does not match rows")
    return rows


def load_covered_experiment(
    gog_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    *,
    enforce_current_counts: bool = True,
) -> CoveredExperiment:
    """Audit source splits, then make train-only encoder context and ROOT queries."""
    paths = {"gog": Path(gog_path), "train": Path(train_path),
             "validation": Path(validation_path), "test": Path(test_path)}
    hashes = {name: _sha256(path) for name, path in paths.items()}
    source = {split: _read_split(paths[split], split) for split in EXPECTED_SOURCE}
    if enforce_current_counts:
        for split, expected in EXPECTED_SOURCE.items():
            if len(source[split]) != expected:
                raise ValueError(f"{split}: expected {expected} source facts, got {len(source[split])}")

    seen_ids: set[str] = set()
    pair_split: dict[tuple[str, str], str] = {}
    train_entities: set[str] = set()
    train_labels: set[str] = set()
    train_keys: set[tuple[Any, ...]] = set()
    for split, rows in source.items():
        for row in rows:
            fid = str(row["fact_id"])
            if fid in seen_ids:
                raise ValueError(f"fact ID appears in multiple source rows: {fid}")
            seen_ids.add(fid)
            head, tail = str(row["head"]), str(row["tail"])
            pair = _pair(head, tail)
            earlier = pair_split.setdefault(pair, split)
            if earlier != split:
                raise ValueError(f"entity pair crosses splits: {pair}")
            if split == "train":
                train_entities.update((head, tail))
                train_labels.add(_label(row))
                train_keys.add(_source_key(head, tail, row["relation"], row.get("display_relation") or row["relation"]))
    for split in ("validation", "test"):
        for row in source[split]:
            if str(row["head"]) not in train_entities or str(row["tail"]) not in train_entities:
                raise ValueError(f"{split} fact {row['fact_id']} has an entity without training support")
            if _label(row) not in train_labels:
                raise ValueError(f"{split} fact {row['fact_id']} has a relation without training support")

    raw = json.loads(paths["gog"].read_text(encoding="utf-8"))
    if len(raw["relations"]) != 3345 and enforce_current_counts:
        raise ValueError("current GoG should contain exactly 3,345 context relations")
    if any(row.get("split") not in (None, "", "train") for row in raw["relations"]):
        raise ValueError("GoG context includes a held-out split label")
    root_graphs = [g for g in raw["graphs"] if g.get("type") == "root"]
    if len(root_graphs) != 1:
        raise ValueError("expected one ROOT graph")
    root_id = root_graphs[0]["graph_id"]
    root_ids = {str(n["entity_id"]) for n in raw["graph_nodes"] if n["graph_id"] == root_id}
    if enforce_current_counts and len(root_ids) != 624:
        raise ValueError(f"expected 624 ROOT entities, got {len(root_ids)}")
    node_by_id = {n["node_id"]: n for n in raw["graph_nodes"]}
    for relation in raw["relations"]:
        head = node_by_id[relation["source_node_id"]]["entity_id"]
        tail = node_by_id[relation["target_node_id"]]["entity_id"]
        key = _source_key(head, tail, relation["type"], relation.get("display_relation") or relation["type"])
        if key not in train_keys:
            raise ValueError(f"GoG relation {relation['relation_id']} is not backed by a training fact")

    # Explicit train labels suppress the legacy loader's automatic GoG re-split.
    context_raw = dict(raw, relations=[dict(row, split="train") for row in raw["relations"]])
    gog = preprocess_data(context_raw)
    if gog.report["splits_generated"] or any(f["split"] != "train" for f in gog.facts):
        raise ValueError("GoG context was unexpectedly re-split")
    flat_raw = dict(raw,
                    graphs=root_graphs,
                    graph_nodes=[n for n in raw["graph_nodes"] if n["graph_id"] == root_id],
                    relations=[dict(r, split="train") for r in raw["relations"] if r["scope"] == "root"])
    flat = preprocess_data(flat_raw)
    entity_ids = [str(e["entity_id"]) for e in gog.entities]
    if entity_ids != [str(e["entity_id"]) for e in flat.entities]:
        raise ValueError("flat and GoG entity mappings differ")
    entity_index = {eid: i for i, eid in enumerate(entity_ids)}
    for data in (gog, flat):
        if any(i < 0 for i in data.root_occurrence_by_entity):
            raise ValueError("an exported entity lacks a ROOT occurrence")
    relation_labels = [label for label, _ in sorted(gog.relation_id_to_index.items(), key=lambda item: item[1])]
    if set(relation_labels) != set(flat.relation_id_to_index):
        raise ValueError("flat and GoG relation vocabularies differ")

    covered_rows = {split: [row for row in rows
                            if str(row["head"]) in root_ids and str(row["tail"]) in root_ids]
                    for split, rows in source.items()}
    if enforce_current_counts:
        for split, expected in EXPECTED_COVERED.items():
            if len(covered_rows[split]) != expected:
                raise ValueError(f"{split}: expected {expected} ROOT-covered facts, got {len(covered_rows[split])}")
    allowed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in covered_rows["train"]:
        label = _label(row)
        allowed[(label, "head")].add(str(row["head_type"]))
        allowed[(label, "tail")].add(str(row["tail_type"]))
    facts: dict[str, list[PilotFact]] = {}
    for split, rows in covered_rows.items():
        converted = []
        for row in rows:
            label = _label(row)
            if label not in gog.relation_id_to_index:
                raise ValueError(f"covered fact {row['fact_id']} has no GoG relation embedding")
            if (str(row["head_type"]) not in allowed[(label, "head")] or
                    str(row["tail_type"]) not in allowed[(label, "tail")]):
                raise ValueError(f"covered {split} fact {row['fact_id']} violates training-derived type rules")
            head, tail = str(row["head"]), str(row["tail"])
            converted.append(PilotFact(str(row["fact_id"]), entity_index[head],
                                       gog.relation_id_to_index[label], entity_index[tail],
                                       label, _pair(head, tail)))
        facts[split] = converted
    type_by_entity = {str(e["entity_id"]): str(e["type_id"]) for e in gog.entities}
    typed_candidates: dict[tuple[int, str], tuple[int, ...]] = {}
    for (label, side), types in allowed.items():
        rid = gog.relation_id_to_index[label]
        typed_candidates[(rid, side)] = tuple(i for i, eid in enumerate(entity_ids)
                                               if type_by_entity[eid] in types)
    for split in ("validation", "test"):
        for fact in facts[split]:
            if (fact.head not in typed_candidates[(fact.relation, "head")] or
                    fact.tail not in typed_candidates[(fact.relation, "tail")]):
                raise ValueError(f"{split} fact {fact.fact_id} has no typed candidate")

    root_relations = [r for r in raw["relations"] if r["scope"] == "root"]
    if enforce_current_counts and len(root_relations) != 1468:
        raise ValueError("expected 1,468 ROOT context links")
    covered_training_keys = {
        _source_key(row["head"], row["tail"], row["relation"],
                    row.get("display_relation") or row["relation"])
        for row in covered_rows["train"]
    }
    root_context_keys = {
        _source_key(node_by_id[row["source_node_id"]]["entity_id"],
                    node_by_id[row["target_node_id"]]["entity_id"],
                    row["type"], row.get("display_relation") or row["type"])
        for row in root_relations
    }
    if root_context_keys != covered_training_keys:
        raise ValueError("ROOT context does not represent exactly the covered training links")
    audit = {
        "source_counts": {s: len(rows) for s, rows in source.items()},
        "covered_counts": {s: len(rows) for s, rows in covered_rows.items()},
        "covered_pair_counts": {s: len({f.pair for f in rows}) for s, rows in facts.items()},
        "root_entities": len(root_ids),
        "context_relations": len(raw["relations"]),
        "root_context_relations": len(root_relations),
        "relation_labels": len(relation_labels),
        "train_derived_type_rules": {
            label: {side: sorted(allowed[(label, side)]) for side in ("head", "tail")}
            for label in sorted({key[0] for key in allowed})
        },
        "test_relation_mix": dict(sorted(Counter(r["relation"] for r in covered_rows["test"]).items())),
        "split_pair_overlap": 0,
        "context_held_out_pair_overlap": 0,
        "automatic_gog_split": False,
    }
    return CoveredExperiment(gog, flat, facts, entity_ids, relation_labels,
                             typed_candidates, hashes, audit)


class FlattenedContextRGCN(nn.Module):
    """Flat R-GCN over every GoG occurrence plus explicit structural edges."""

    def __init__(self, data: PreprocessedGoG, hidden_dim: int, num_bases: int, dropout: float):
        super().__init__()
        try:
            from torch_geometric.nn import RGCNConv
        except ImportError as exc:
            raise RuntimeError("Install torch_geometric for the flattened context control") from exc
        self.initializer = Stage1Initializer(len(data.entities), len(data.type_id_to_index), hidden_dim)
        self.graph_token = nn.Parameter(torch.zeros(hidden_dim))
        relation_count = len(data.relation_id_to_index)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, 2 * relation_count + 6,
                              num_bases=num_bases, root_weight=True)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, 2 * relation_count + 6,
                              num_bases=num_bases, root_weight=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_occurrences = len(data.graph_nodes)
        internal = sorted((g for g in data.graphs if g["kind"] == "internal"), key=lambda g: g["graph_id"])
        containers = {g["graph_id"]: self.n_occurrences + i for i, g in enumerate(internal)}
        self.register_buffer("owner_entity", torch.tensor(
            [data.entity_id_to_index[g["owner_entity_id"]] for g in internal], dtype=torch.long))

        source: list[int] = []
        target: list[int] = []
        edge_type: list[int] = []

        def add_edge(a: int, b: int, forward: int, reverse: int) -> None:
            source.extend((a, b))
            target.extend((b, a))
            edge_type.extend((forward, reverse))

        for edges in data.edge_lists.values():
            for edge in edges:
                add_edge(edge.source_occurrence_index, edge.target_occurrence_index,
                         edge.relation_index, edge.relation_index + relation_count)
        for i, node in enumerate(data.graph_nodes):
            graph_id = node["graph_id"]
            if graph_id not in containers:
                continue
            entity = data.entity_id_to_index[node["entity_id"]]
            add_edge(data.root_occurrence_by_entity[entity], i,
                     2 * relation_count, 2 * relation_count + 1)
            add_edge(i, containers[graph_id],
                     2 * relation_count + 2, 2 * relation_count + 3)
        for graph in internal:
            owner = data.entity_id_to_index[graph["owner_entity_id"]]
            add_edge(data.root_occurrence_by_entity[owner], containers[graph["graph_id"]],
                     2 * relation_count + 4, 2 * relation_count + 5)
        self.register_buffer("edge_index", torch.tensor([source, target], dtype=torch.long))
        self.register_buffer("edge_type", torch.tensor(edge_type, dtype=torch.long))

    def forward(self, data: PreprocessedGoG) -> Tensor:
        device = self.initializer.entity_embedding.weight.device
        types = torch.tensor(data.type_index_by_entity, dtype=torch.long, device=device)
        occurrences = torch.tensor(data.entity_index_by_occurrence, dtype=torch.long, device=device)
        base, _, z = self.initializer(types, occurrences)
        containers = base[self.owner_entity] + self.graph_token
        z = torch.cat((z, containers), dim=0)
        z = self.norm1(z + self.dropout(F.relu(self.conv1(z, self.edge_index, self.edge_type))))
        z = self.norm2(z + self.dropout(F.relu(self.conv2(z, self.edge_index, self.edge_type))))
        return z[:self.n_occurrences]


def _known_sets(facts: dict[str, list[PilotFact]], splits: tuple[str, ...]) -> tuple[set[tuple[int, int, int]], dict[tuple[str, int, int], set[int]]]:
    triples: set[tuple[int, int, int]] = set()
    by_query: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for split in splits:
        for fact in facts[split]:
            triples.add((fact.head, fact.relation, fact.tail))
            by_query[("head", fact.relation, fact.tail)].add(fact.head)
            by_query[("tail", fact.relation, fact.head)].add(fact.tail)
    return triples, by_query


def _negative_options(experiment: CoveredExperiment) -> list[dict[str, tuple[int, ...]]]:
    known, _ = _known_sets(experiment.facts, ("train",))
    options = []
    for fact in experiment.facts["train"]:
        heads = tuple(candidate for candidate in experiment.typed_candidates[(fact.relation, "head")]
                      if candidate != fact.head and (candidate, fact.relation, fact.tail) not in known)
        tails = tuple(candidate for candidate in experiment.typed_candidates[(fact.relation, "tail")]
                      if candidate != fact.tail and (fact.head, fact.relation, candidate) not in known)
        options.append({"head": heads, "tail": tails})
    return options


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no scorable evaluation queries")
    scores = [1.0 / row["rank"] for row in rows]
    result: dict[str, Any] = {
        "queries": len(rows), "facts": len({r["fact_id"] for r in rows}),
        "pairs": len({tuple(r["pair"]) for r in rows}),
        "mrr": statistics.mean(scores),
        "hits_at_1": statistics.mean(r["rank"] <= 1 for r in rows),
        "hits_at_3": statistics.mean(r["rank"] <= 3 for r in rows),
        "hits_at_10": statistics.mean(r["rank"] <= 10 for r in rows),
        "mean_candidates": statistics.mean(r["candidates"] for r in rows),
    }
    relation_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relation_rows[row["relation"]].append(row)
    result["by_relation"] = {
        relation: {"queries": len(group), "facts": len({r["fact_id"] for r in group}),
                   "mrr": statistics.mean(1.0 / r["rank"] for r in group)}
        for relation, group in sorted(relation_rows.items())
    }
    return result


@torch.no_grad()
def evaluate_fixed(
    experiment: CoveredExperiment,
    decoder: ComplExDecoder,
    occurrence_states: Tensor,
    data: PreprocessedGoG,
    split: str,
    *,
    typed: bool,
) -> dict[str, Any]:
    root_index = torch.tensor(data.root_occurrence_by_entity, dtype=torch.long,
                              device=occurrence_states.device)
    projected = decoder.project_for_ranking(occurrence_states[root_index])
    _, known = _known_sets(experiment.facts, ("train", "validation", "test"))
    groups: dict[tuple[int, str], list[PilotFact]] = defaultdict(list)
    for fact in experiment.facts[split]:
        groups[(fact.relation, "head")].append(fact)
        groups[(fact.relation, "tail")].append(fact)
    rows: list[dict[str, Any]] = []
    for (relation, side), queries in sorted(groups.items()):
        pool = (experiment.typed_candidates[(relation, side)] if typed
                else tuple(range(len(experiment.entity_ids))))
        positions = {candidate: i for i, candidate in enumerate(pool)}
        fixed = [fact.tail if side == "head" else fact.head for fact in queries]
        targets = [fact.head if side == "head" else fact.tail for fact in queries]
        if any(target not in positions for target in targets):
            raise ValueError("true endpoint missing from evaluation candidates")
        scores = decoder.score_candidate_matrix(
            projected,
            torch.tensor(fixed, dtype=torch.long, device=projected.device),
            torch.full((len(queries),), relation, dtype=torch.long, device=projected.device),
            torch.tensor(pool, dtype=torch.long, device=projected.device), side).cpu()
        for i, fact in enumerate(queries):
            positive_score = scores[i, positions[targets[i]]]
            allowed = scores[i] > positive_score
            for other in known.get((side, relation, fixed[i]), ()):
                if other != targets[i] and other in positions:
                    allowed[positions[other]] = False
            rank = 1 + int(allowed.sum())
            rows.append({"fact_id": fact.fact_id, "pair": list(fact.pair),
                         "relation": fact.relation_label, "side": side,
                         "rank": rank, "candidates": len(pool)})
    return {"summary": _score_summary(rows), "queries": rows,
            "candidate_mode": "train_type_aware" if typed else "all_root",
            "ranking": "filtered; optimistic ties; known positives from all source splits used only during evaluation"}


def _build_encoder(kind: str, experiment: CoveredExperiment, config: TrainingConfig) -> nn.Module:
    if kind == "flat_root":
        return RGCNBaseline.from_preprocessed(experiment.flat, hidden_dim=config.hidden_dim,
                                               num_bases=config.num_bases, dropout=config.dropout)
    if kind == "flat_context":
        return FlattenedContextRGCN(experiment.gog, config.hidden_dim,
                                    config.num_bases, config.dropout)
    if kind == "gog":
        return GoGEncoder.from_preprocessed(experiment.gog, hidden_dim=config.hidden_dim,
                                            num_bases=config.num_bases, num_heads=config.num_heads,
                                            dropout=config.dropout, num_rounds=config.num_rounds)
    raise ValueError(f"unknown covered-pilot model {kind}")


def _encode(kind: str, encoder: nn.Module, data: PreprocessedGoG) -> Tensor:
    if kind == "gog":
        return encoder(data).z_occurrence
    if kind == "flat_root":
        return encoder(data)[1]
    return encoder(data)


def _training_rows(experiment: CoveredExperiment, data: PreprocessedGoG) -> list[dict[str, str]]:
    root_node_ids = [data.graph_nodes[index]["node_id"] for index in data.root_occurrence_by_entity]
    return [{"fact_id": f.fact_id, "source_node_id": root_node_ids[f.head],
             "target_node_id": root_node_ids[f.tail], "relation_type_id": f.relation_label}
            for f in experiment.facts["train"]]


def train_one(
    experiment: CoveredExperiment,
    kind: str,
    config: TrainingConfig,
    seed: int,
    output_dir: Path,
    negative_options: list[dict[str, tuple[int, ...]]],
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(config.device)
    data = experiment.flat if kind == "flat_root" else experiment.gog
    encoder = _build_encoder(kind, experiment, config).to(device)
    decoder = ComplExDecoder(len(experiment.relation_labels), config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    rows = _training_rows(experiment, data)
    root_occ = data.root_occurrence_by_entity
    heldout_triples = {(fact.head, fact.relation, fact.tail)
                       for split in ("validation", "test") for fact in experiment.facts[split]}
    best_score = -float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.epochs):
        rng = random.Random(seed + epoch * 1009)
        negatives = []
        shortfalls = {"head": 0, "tail": 0}
        heldout_negative_collisions = 0
        for i, options in enumerate(negative_options):
            positive = experiment.facts["train"][i]
            for side in ("head", "tail"):
                pool = options[side]
                if len(pool) < config.negatives_per_side:
                    shortfalls[side] += 1
                for entity_index in rng.sample(pool, min(config.negatives_per_side, len(pool))):
                    negatives.append((i, side, root_occ[entity_index]))
                    sampled = ((entity_index, positive.relation, positive.tail) if side == "head"
                               else (positive.head, positive.relation, entity_index))
                    heldout_negative_collisions += sampled in heldout_triples
        encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        occurrence_states = _encode(kind, encoder, data)
        loss = _backward_in_fact_batches(data, decoder, occurrence_states, negatives, rows,
                                         data.node_id_to_index, experiment.gog.relation_id_to_index,
                                         config.training_batch_size)
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(f"non-finite loss in {kind}, seed {seed}, epoch {epoch + 1}")
        nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], config.gradient_clip)
        optimizer.step()
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            states = _encode(kind, encoder, data)
            validation = evaluate_fixed(experiment, decoder, states, data, "validation", typed=True)
        score = validation["summary"]["mrr"]
        history.append({"epoch": epoch + 1, "loss": loss, "validation_typed_mrr": score,
                        "negative_shortfalls": shortfalls,
                        "heldout_negative_collisions_audit_only": heldout_negative_collisions})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = {"encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                          "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}}
        else:
            stale += 1
        if config.show_progress:
            print(f"[{kind} seed={seed}] epoch={epoch + 1} loss={loss:.4f} "
                  f"validation_typed_MRR={score:.4f} best={best_epoch}", flush=True)
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        states = _encode(kind, encoder, data)
        validation_typed = evaluate_fixed(experiment, decoder, states, data, "validation", typed=True)
        validation_all = evaluate_fixed(experiment, decoder, states, data, "validation", typed=False)
        test_typed = evaluate_fixed(experiment, decoder, states, data, "test", typed=True)
        test_all = evaluate_fixed(experiment, decoder, states, data, "test", typed=False)
    checkpoint_path = output_dir / f"{kind}_seed{seed}_best.pt"
    torch.save({"encoder": best_state["encoder"], "decoder": best_state["decoder"],
                "kind": kind, "seed": seed, "best_epoch": best_epoch,
                "config": asdict(config), "input_sha256": experiment.hashes,
                "entity_ids": experiment.entity_ids, "relation_labels": experiment.relation_labels}, checkpoint_path)
    return {"model": kind, "seed": seed, "best_epoch": best_epoch,
            "parameter_count": sum(p.numel() for p in encoder.parameters()) +
                               sum(p.numel() for p in decoder.parameters()),
            "elapsed_seconds": time.time() - start_time, "history": history,
            "validation": {"typed": validation_typed, "all_root": validation_all},
            "test": {"typed": test_typed, "all_root": test_all},
            "checkpoint": str(checkpoint_path)}


def _pair_scores(run: dict[str, Any]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in run["test"]["typed"]["queries"]:
        values[tuple(row["pair"])].append(1.0 / row["rank"])
    return {pair: statistics.mean(scores) for pair, scores in values.items()}


def _paired_bootstrap(runs: dict[str, list[dict[str, Any]]], first: str, second: str) -> dict[str, Any]:
    per_model: dict[str, dict[tuple[str, str], list[float]]] = {}
    for kind in (first, second):
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for run in runs[kind]:
            for pair, score in _pair_scores(run).items():
                values[pair].append(score)
        per_model[kind] = values
    pairs = sorted(per_model[first])
    if pairs != sorted(per_model[second]):
        raise ValueError("models were evaluated on different test pairs")
    differences = [statistics.mean(per_model[first][pair]) -
                   statistics.mean(per_model[second][pair]) for pair in pairs]
    rng = random.Random(42)
    draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return {"first": first, "second": second, "mean_typed_mrr_difference": statistics.mean(differences),
            "pair_bootstrap_95_percent_interval": [draws[49], draws[1949]],
            "independent_entity_pairs": len(pairs), "bootstrap_samples": 2000}


def run_covered_pilot(
    gog_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    seeds: tuple[int, ...] = (17, 23, 42, 73, 101),
    *,
    enforce_current_counts: bool = True,
) -> dict[str, Any]:
    experiment = load_covered_experiment(gog_path, train_path, validation_path, test_path,
                                         enforce_current_counts=enforce_current_counts)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    options = _negative_options(experiment)
    runs: dict[str, list[dict[str, Any]]] = {kind: [] for kind in MODEL_KINDS}
    for seed in seeds:
        for kind in MODEL_KINDS:
            run = train_one(experiment, kind, config, seed, output, options)
            runs[kind].append(run)
            (output / f"{kind}_seed{seed}_report.json").write_text(
                json.dumps(run, indent=2), encoding="utf-8")
    comparisons = [_paired_bootstrap(runs, "gog", "flat_root"),
                   _paired_bootstrap(runs, "gog", "flat_context")]
    summary = {
        "notice": "Covered-entity pilot only; not a full PrimeKG benchmark result.",
        "input_sha256": experiment.hashes,
        "audit": experiment.audit,
        "settings": {"config": asdict(config), "seeds": list(seeds),
                     "primary": "train-type-aware filtered MRR",
                     "secondary": "all-ROOT filtered MRR and Hits@1/3/10",
                     "negative_filter": "training positives only",
                     "evaluation_filter": "known positives from train, validation and test; evaluation only",
                     "decoder": "ComplEx shared architecture; independent weights per model and seed"},
        "models": {kind: [{"seed": run["seed"], "best_epoch": run["best_epoch"],
                           "parameter_count": run["parameter_count"],
                           "validation_typed": run["validation"]["typed"]["summary"],
                           "test_typed": run["test"]["typed"]["summary"],
                           "test_all_root": run["test"]["all_root"]["summary"],
                           "report": f"{kind}_seed{run['seed']}_report.json",
                           "checkpoint": run["checkpoint"]} for run in model_runs]
                   for kind, model_runs in runs.items()},
        "model_aggregates": {
            kind: {
                "mean_test_typed_mrr": statistics.mean(run["test"]["typed"]["summary"]["mrr"]
                                                       for run in model_runs),
                "mean_test_all_root_mrr": statistics.mean(run["test"]["all_root"]["summary"]["mrr"]
                                                          for run in model_runs),
                "test_typed_mrr_by_seed": {str(run["seed"]): run["test"]["typed"]["summary"]["mrr"]
                                           for run in model_runs},
                "audited_heldout_negative_collisions": sum(
                    epoch["heldout_negative_collisions_audit_only"]
                    for run in model_runs for epoch in run["history"]),
            }
            for kind, model_runs in runs.items()
        },
        "paired_comparisons": comparisons,
        "evaluated_fact_ids": {s: [f.fact_id for f in experiment.facts[s]]
                               for s in ("validation", "test")},
    }
    (output / "covered_pilot_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
