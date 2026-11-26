"""Train link predictors on a GoG dataset."""

import argparse
import json
from pathlib import Path

from data_pipeline.preprocessing import load_gog
from training.link_prediction import TrainingConfig, run_baseline, run_comparison, run_gog, run_rgcn


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
    args = parser.parse_args()
    config = TrainingConfig(hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
                            negatives_per_side=args.negatives, learning_rate=args.learning_rate,
                            training_batch_size=args.batch_size,
                            seed=args.seed, device=args.device, show_progress=True)
    data = load_gog(args.dataset, split_seed=args.split_seed)
    if args.model == "baseline":
        result = run_baseline(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "baseline_test": result["baseline"]["test"],
                          "report": str(args.output_dir / "baseline_report.json")}, indent=2))
    elif args.model == "gog":
        result = run_gog(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "gog_test": result["gog"]["test"],
                          "report": str(args.output_dir / "gog_report.json")}, indent=2))
    elif args.model == "rgcn":
        result = run_rgcn(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "rgcn_test": result["rgcn"]["test"],
                          "report": str(args.output_dir / "rgcn_report.json")}, indent=2))
    else:
        result = run_comparison(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "baseline_test": result["baseline"]["test"],
                          "gog_test": result["gog"]["test"],
                          "report": str(args.output_dir / "comparison_report.json")}, indent=2))


if __name__ == "__main__":
    main()
