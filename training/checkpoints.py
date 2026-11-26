"""ID-aligned baseline transfer; loading never assumes table row alignment."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from data_pipeline.preprocessing import PreprocessedGoG
from gog_embedding.structural import ELEMENTS, FRAGMENTS


def initialize_from_checkpoint(encoder: nn.Module, decoder: nn.Module, data: PreprocessedGoG,
                               path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = checkpoint["encoder"]
    if checkpoint["decoder"]["projection.weight"].shape != decoder.projection.weight.shape:
        raise ValueError("checkpoint hidden dimension does not match model")
    if checkpoint.get("kind") != "baseline":
        vocabularies = checkpoint.get("feature_vocabularies")
        if vocabularies is not None and vocabularies != {"elements": ELEMENTS, "fragments": FRAGMENTS}:
            raise ValueError("structural checkpoint feature vocabularies do not match model")
        # Historical GoG checkpoints remain strict-loadable in the legacy profile.
        for key, mapping in (("entity_id_to_index", data.entity_id_to_index),
                             ("relation_id_to_index", data.relation_id_to_index),
                             ("node_id_to_index", data.node_id_to_index),
                             ("graph_id_to_index", data.graph_id_to_index)):
            if checkpoint.get(key) != mapping:
                raise ValueError(f"exact checkpoint restore requires identical {key}")
        if checkpoint.get("type_id_to_index", data.type_id_to_index) != data.type_id_to_index:
            raise ValueError("exact checkpoint restore requires identical type mapping")
        encoder.load_state_dict(source)
        decoder.load_state_dict(checkpoint["decoder"])
        return {"path": str(path), "mode": "exact", "source_best_epoch": checkpoint.get("best_epoch")}

    root_graph = next(g["graph_id"] for g in data.graphs if g["kind"] == "root")
    root_nodes = {n["node_id"] for n in data.graph_nodes if n["graph_id"] == root_graph}
    required_entities = {n["entity_id"] for n in data.graph_nodes if n["node_id"] in root_nodes}
    required_types = {data.entities[data.entity_id_to_index[e]]["type_id"] for e in required_entities}
    required_relations = {f["relation_type_id"] for f in data.facts
                          if f["source_node_id"] in root_nodes and f["target_node_id"] in root_nodes}
    type_map = checkpoint.get("type_id_to_index")
    if type_map is None:
        if checkpoint.get("entity_id_to_index") != data.entity_id_to_index:
            raise ValueError("legacy baseline transfer across datasets requires a type_id_to_index mapping")
        type_map = data.type_id_to_index

    def copy_rows(target: torch.Tensor, old: torch.Tensor, target_map: dict,
                  source_map: dict, required: set, label: str) -> int:
        missing = required - source_map.keys()
        if missing:
            raise ValueError(f"checkpoint is missing required ROOT {label}: {sorted(missing)[:3]}")
        common = sorted(target_map.keys() & source_map.keys())
        target_rows = torch.tensor([target_map[k] for k in common], device=target.device)
        old_rows = torch.tensor([source_map[k] for k in common], dtype=torch.long)
        target.index_copy_(0, target_rows.long(), old[old_rows].to(target.device))
        return len(common)

    with torch.no_grad():
        entities = copy_rows(encoder.initializer.entity_embedding.weight,
                             source["initializer.entity_embedding.weight"], data.entity_id_to_index,
                             checkpoint["entity_id_to_index"], required_entities, "entities")
        types = copy_rows(encoder.initializer.type_embedding.weight,
                          source["initializer.type_embedding.weight"], data.type_id_to_index,
                          type_map, required_types, "types")
        encoder.initializer.occurrence_norm.load_state_dict({
            k.removeprefix("initializer.occurrence_norm."): v for k, v in source.items()
            if k.startswith("initializer.occurrence_norm.")})
        decoder.projection.load_state_dict({k.removeprefix("projection."): v
                                           for k, v in checkpoint["decoder"].items()
                                           if k.startswith("projection.")})
        relations = copy_rows(decoder.relation.weight, checkpoint["decoder"]["relation.weight"],
                              data.relation_id_to_index, checkpoint["relation_id_to_index"],
                              required_relations, "relations")
    return {"path": str(path), "mode": "baseline_transfer", "entities_copied": entities,
            "types_copied": types, "relations_copied": relations,
            "source_best_epoch": checkpoint.get("best_epoch"), "optimizer_reset": True}
