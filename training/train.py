"""Train link predictors on a GoG dataset."""

import argparse
import json
from pathlib import Path

from data_pipeline.preprocessing import load_gog
from training.link_prediction import TrainingConfig, run_baseline, run_comparison, run_gog, run_rgcn


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-profile", choices=("legacy", "structural"), default="legacy")
    parser.add_argument("--task-scope", choices=("all", "root"), default=None)
    parser.add_argument("--feature-mode", choices=("auto", "id_type", "categorical"), default="auto")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--num-bases", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--basis-init", choices=("legacy", "per_matrix"), default=None)
    parser.add_argument("--composition", choices=("multiply", "residual"), default=None)
    parser.add_argument("--normalization", choices=("global", "relation"), default=None)
    parser.add_argument("--embedding-learning-rate", type=float, default=None)
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--decoder-learning-rate", type=float, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--update-offset", type=int, default=0,
                        help="completed updates used to offset the negative-sampling schedule")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--diagnostic-train-limit", type=int, default=32)
    for flag in ("identity-residual", "local", "cross", "fusion", "root", "family-readout"):
        parser.add_argument("--" + flag, action=argparse.BooleanOptionalAction, default=True)


def model_config_options(args: argparse.Namespace) -> dict:
    return dict(encoder_profile=args.encoder_profile,
                task_scope=args.task_scope or ("root" if args.encoder_profile == "structural" else "all"),
                feature_mode=args.feature_mode, num_rounds=args.rounds, num_bases=args.num_bases,
                num_heads=args.num_heads, dropout=args.dropout, weight_decay=args.weight_decay,
                gradient_clip=args.gradient_clip, basis_init=args.basis_init, composition=args.composition,
                normalization=args.normalization, identity_residual=args.identity_residual,
                local_enabled=args.local, cross_enabled=args.cross, fusion_enabled=args.fusion,
                root_enabled=args.root, family_readout=args.family_readout,
                sync_enabled=args.sync, feedback_enabled=args.feedback,
                embedding_learning_rate=args.embedding_learning_rate,
                encoder_learning_rate=args.encoder_learning_rate,
                decoder_learning_rate=args.decoder_learning_rate, init_checkpoint=args.init_checkpoint,
                update_offset=args.update_offset,
                diagnostic_train_limit=args.diagnostic_train_limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="GoG JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/link-prediction"))
    parser.add_argument("--model", choices=("baseline", "rgcn", "gog", "compare"),
                        default="baseline",
                        help="train entity-only, R-GCN, GoG, or prototype comparison")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="training facts per decoder batch; lower this if GPU memory is limited")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for deterministic splits when the input has no split labels")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda")
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed-budget", action="store_true", help="disable early stopping")
    add_model_arguments(parser)
    args = parser.parse_args()
    config = TrainingConfig(hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
                            negatives_per_side=args.negatives, learning_rate=args.learning_rate,
                            training_batch_size=args.batch_size,
                            seed=args.seed, device=args.device, show_progress=True,
                            early_stopping=not args.fixed_budget,
                            evaluate_test=(args.encoder_profile == "legacy" if args.evaluate_test is None
                                           else args.evaluate_test), **model_config_options(args))
    data = load_gog(args.dataset, split_seed=args.split_seed)
    if args.model == "baseline":
        result = run_baseline(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "baseline_test": result["baseline"]["test"],
                          "baseline_validation": result["baseline"]["validation"]["aggregate"],
                          "report": str(args.output_dir / "baseline_report.json")}, indent=2))
    elif args.model == "gog":
        result = run_gog(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "gog_test": result["gog"]["test"],
                          "gog_validation": result["gog"]["validation"]["aggregate"],
                          "report": str(args.output_dir / "gog_report.json")}, indent=2))
    elif args.model == "rgcn":
        result = run_rgcn(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "rgcn_test": result["rgcn"]["test"],
                          "rgcn_validation": result["rgcn"]["validation"]["aggregate"],
                          "report": str(args.output_dir / "rgcn_report.json")}, indent=2))
    else:
        result = run_comparison(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "baseline_test": result["baseline"]["test"],
                          "gog_test": result["gog"]["test"],
                          "report": str(args.output_dir / "comparison_report.json")}, indent=2))


if __name__ == "__main__":
    main()
