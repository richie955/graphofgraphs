"""Train the entity-only baseline, or compare it with the full GoG encoder."""

import argparse
import json
from pathlib import Path

from gog_preprocessing import load_gog
from stage9_training import TrainingConfig, run_baseline, run_comparison, run_gog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="GoG JSON file (including docs/primekg_gog_30000.json)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/stage9"))
    parser.add_argument("--model", choices=("baseline", "gog", "compare"), default="baseline",
                        help="train baseline only, GoG only, or both for comparison")
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
    else:
        result = run_comparison(data, config, args.output_dir)
        print(json.dumps({"notice": result["notice"],
                          "baseline_test": result["baseline"]["test"],
                          "gog_test": result["gog"]["test"],
                          "report": str(args.output_dir / "comparison_report.json")}, indent=2))


if __name__ == "__main__":
    main()
