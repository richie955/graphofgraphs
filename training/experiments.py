"""Validation-first, fixed-budget comparisons on one frozen GoG dataset."""

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from data_pipeline.preprocessing import load_gog
from .link_prediction import TrainingConfig, _run_one, protocol_fingerprints


ABLATIONS = {
    "identity_off": {"identity_residual": False},
    "chemical_off": {"feature_mode": "id_type"},
    "message_legacy": {"basis_init": "legacy", "composition": "multiply", "normalization": "global"},
    "basis_legacy": {"basis_init": "legacy"},
    "composition_multiply": {"composition": "multiply"},
    "normalization_global": {"normalization": "global"},
    "readout_shared": {"family_readout": False},
    "fusion_off": {"fusion_enabled": False},
    "local_off": {"local_enabled": False},
    "cross_off": {"cross_enabled": False},
    "root_off": {"root_enabled": False},
    "feedback": {"num_rounds": 2, "feedback_enabled": True, "sync_enabled": False},
    "sync_feedback": {"num_rounds": 2, "feedback_enabled": True, "sync_enabled": True},
}


def run_suite(data, output_dir: str | Path, config: TrainingConfig | None = None,
              seeds=(17, 23, 42), warmup_updates: int = 80, continuation_updates: int = 80,
              ablations=(), evaluate_test: bool = False) -> dict:
    if warmup_updates < 1 or continuation_updates < 1:
        raise ValueError("both phase budgets must be positive")
    seeds, ablations = tuple(seeds), tuple(ablations)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    if len(set(ablations)) != len(ablations) or any(a not in ABLATIONS for a in ablations):
        raise ValueError("provide distinct supported ablations")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = config or TrainingConfig(hidden_dim=128, negatives_per_side=50,
                                    encoder_profile="structural", task_scope="root", evaluate_test=False)
    if base.init_checkpoint is not None:
        raise ValueError("the suite owns baseline pretraining and checkpoint initialization")
    base = replace(base, encoder_profile="structural", task_scope="root", early_stopping=False, init_checkpoint=None,
                   evaluate_test=evaluate_test)
    if base.sync_enabled and not base.feedback_enabled:
        raise ValueError("synchronization requires feedback")
    if (base.sync_enabled or base.feedback_enabled) and (base.num_rounds or 1) < 2:
        raise ValueError("refinement requires at least two rounds")
    fingerprints = protocol_fingerprints(data, "root")
    manifest = {"config": asdict(base), "seeds": list(seeds), "warmup_updates": warmup_updates,
                "continuation_updates": continuation_updates, "ablations": list(ablations),
                "fingerprints": fingerprints, "task": "ROOT positives and all ROOT candidates",
                "test_evaluated": base.evaluate_test,
                "note": "All final configurations have the same maximum total update budget; "
                        "warm-start branches share baseline pretraining and reset optimizers. "
                        "Best validation checkpoints may precede the budget endpoint."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runs, rows = [], []
    started = time.perf_counter()
    for seed in seeds:
        directory = output / f"seed_{seed}"
        reference = replace(base, seed=seed, encoder_profile="legacy", feature_mode="id_type",
                            num_rounds=2, basis_init=None, composition=None, normalization=None,
                            local_enabled=True, cross_enabled=True, fusion_enabled=True,
                            root_enabled=True, sync_enabled=False, feedback_enabled=False,
                            identity_residual=False, family_readout=False, update_offset=0)
        warm_config = replace(reference, epochs=warmup_updates, evaluate_test=False)
        warm = _run_one(data, warm_config, "baseline", directory / "pretraining")
        structural = replace(base, seed=seed, encoder_profile="structural", epochs=continuation_updates,
                             init_checkpoint=warm["checkpoint"], update_offset=warmup_updates)
        settings = [
            ("baseline_continuation", "baseline", replace(reference, epochs=continuation_updates,
                 init_checkpoint=warm["checkpoint"], update_offset=warmup_updates)),
            ("rgcn", "rgcn", replace(reference, epochs=warmup_updates + continuation_updates)),
            ("gog_legacy", "gog", replace(reference, epochs=warmup_updates + continuation_updates)),
            ("gog_structural", "gog", structural),
            *[("ablation_" + label, "gog", replace(structural, **ABLATIONS[label])) for label in ablations],
        ]
        seed_run = {"seed": seed, "shared_pretraining": warm, "models": {}}
        for label, kind, model_config in settings:
            report = _run_one(data, model_config, kind, directory / label)
            if report["fingerprints"] != fingerprints:
                raise RuntimeError("comparison protocol changed between models")
            expected = (warm["negative_sample_fingerprints"]
                        if model_config.update_offset == 0 else [])
            if expected and report["negative_sample_fingerprints"][:warmup_updates] != expected:
                raise RuntimeError("pretraining negative schedule differs across models")
            continuation = report["negative_sample_fingerprints"][
                0 if model_config.update_offset else warmup_updates:]
            if seed_run["models"]:
                baseline_negatives = seed_run["models"]["baseline_continuation"]["negative_sample_fingerprints"]
                if continuation != baseline_negatives:
                    raise RuntimeError("continuation negative schedule differs across models")
            report["config"] = asdict(model_config)
            report["total_budget_updates"] = warmup_updates + continuation_updates
            report["shared_pretraining_updates"] = warmup_updates if model_config.init_checkpoint else 0
            seed_run["models"][label] = report
            (directory / label / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            root = report["validation"]["root"]
            row = {"seed": seed, "model": label, "optimizer_updates_this_run": report["optimizer_updates"],
                   "total_budget_updates": report["total_budget_updates"], "best_epoch": report["best_epoch"],
                   "elapsed_seconds": report["elapsed_seconds"]}
            for metric_kind in ("raw", "filtered"):
                for key in ("mrr", "hits@1", "hits@3", "hits@10", "mean_rank"):
                    row[metric_kind + "_" + key] = root[metric_kind][key]
            rows.append(row)
        runs.append(seed_run)
    aggregates = {}
    for label in runs[0]["models"]:
        values = [r["filtered_mrr"] for r in rows if r["model"] == label and r["filtered_mrr"] is not None]
        aggregates[label] = {"validation_filtered_mrr_mean": statistics.mean(values) if values else None,
                             "validation_filtered_mrr_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                             "seeds": len(values)}
    result = {"manifest": manifest, "runs": runs, "aggregate": aggregates,
              "elapsed_seconds": time.perf_counter() - started}
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    from .train import add_model_arguments, model_config_options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/structural-comparison"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42])
    parser.add_argument("--warmup-updates", type=int, default=80)
    parser.add_argument("--continuation-updates", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--ablations", choices=sorted(ABLATIONS), nargs="*", default=[])
    add_model_arguments(parser)
    parser.set_defaults(encoder_profile="structural", task_scope="root")
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    options = model_config_options(args)
    if (options["init_checkpoint"] is not None or options["task_scope"] != "root"
            or options["encoder_profile"] != "structural"):
        parser.error("the suite owns initialization and requires the structural profile and ROOT-only targets")
    config = TrainingConfig(hidden_dim=args.hidden_dim, negatives_per_side=args.negatives,
                            training_batch_size=args.batch_size, learning_rate=args.learning_rate,
                            device=args.device, show_progress=True, evaluate_test=args.evaluate_test, **options)
    result = run_suite(load_gog(args.dataset), args.output_dir, config, args.seeds,
                       args.warmup_updates, args.continuation_updates, args.ablations,
                       evaluate_test=args.evaluate_test)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
