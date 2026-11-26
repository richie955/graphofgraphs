"""Evaluate saved link-prediction weights without any training updates."""

import argparse
import inspect
import json
from pathlib import Path

import torch

from baselines.rgcn import RGCNBaseline
from data_pipeline.preprocessing import load_gog
from gog_embedding import ComplExDecoder, GoGEncoder
from .link_prediction import EntityOnlyBaseline, choose_device, evaluate, protocol_fingerprints


def evaluate_checkpoint(dataset: str | Path, checkpoint_path: str | Path,
                        output: str | Path, device: str = "auto") -> dict:
    data = load_gog(dataset)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config, kind = saved["config"], saved["kind"]
    for key in ("entity_id_to_index", "node_id_to_index", "graph_id_to_index",
                "relation_id_to_index", "type_id_to_index"):
        if key in saved and saved[key] != getattr(data, key):
            raise ValueError(f"Dataset/checkpoint mismatch: {key}")
    task_scope = config.get("task_scope") or ("root" if config.get("encoder_profile") == "structural" else "all")
    fingerprints = protocol_fingerprints(data, task_scope)
    if "fingerprints" in saved and saved["fingerprints"] != fingerprints:
        raise ValueError("Dataset or evaluation protocol differs from the training checkpoint")
    hidden = config["hidden_dim"]
    if kind == "baseline":
        encoder = EntityOnlyBaseline(len(data.entities), len(data.type_id_to_index), hidden)
    elif kind == "rgcn":
        encoder = RGCNBaseline.from_preprocessed(data, hidden_dim=hidden,
                                                num_bases=config.get("num_bases", 8),
                                                dropout=config.get("dropout", 0.2))
    elif kind == "gog":
        parameters = inspect.signature(GoGEncoder.__init__).parameters
        options = {key: value for key, value in config.items() if key in parameters}
        options["profile"] = config.get("encoder_profile", "legacy")
        encoder = GoGEncoder.from_preprocessed(data, **options)
    else:
        raise ValueError(f"Unsupported checkpoint model: {kind}")
    decoder = ComplExDecoder(len(data.relation_id_to_index), hidden)
    encoder.load_state_dict(saved["encoder"], strict=True)
    decoder.load_state_dict(saved["decoder"], strict=True)
    target = choose_device(device)
    encoder.to(target).eval()
    decoder.to(target).eval()
    print(f"Evaluating {kind} checkpoint, selected epoch {saved.get('best_epoch', 'unknown')}; no training.", flush=True)
    with torch.no_grad():
        encoded = encoder(data)
        states = encoded.z_occurrence if kind == "gog" else encoded[1]
        metrics = evaluate(data, decoder, states, "test", {"train", "validation", "test"}, task_scope)
    result = {"kind": kind, "checkpoint": str(checkpoint_path), "best_epoch": saved.get("best_epoch"),
              "fingerprints": fingerprints, "test": metrics, "optimizer_updates": 0}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    result = evaluate_checkpoint(args.dataset, args.checkpoint, args.output, args.device)
    print(json.dumps(result["test"]["root"], indent=2))
    print("Saved test results:", args.output)


if __name__ == "__main__":
    main()
