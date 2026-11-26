import tempfile
import unittest
from pathlib import Path

import torch

from data_pipeline.preprocessing import load_gog
from training.evaluate_checkpoint import evaluate_checkpoint
from training.link_prediction import TrainingConfig, _run_one


class CheckpointEvaluationTests(unittest.TestCase):
    def test_test_only_evaluation_matches_direct_metrics_without_changing_weights(self):
        torch.set_num_threads(2)
        data_path = Path(__file__).parent.parent / "docs/synthetic_gog_fixture.json"
        data = load_gog(data_path)
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("baseline", "rgcn", "gog"):
                with self.subTest(kind=kind):
                    folder = Path(tmp) / kind
                    result = _run_one(data, TrainingConfig(hidden_dim=16, num_bases=2, num_heads=4,
                                      epochs=1, dropout=0, task_scope="root", device="cpu",
                                      evaluate_test=True), kind, folder)
                    path = Path(result["checkpoint"])
                    before = path.read_bytes()
                    evaluated = evaluate_checkpoint(data_path, path, folder / "test.json", "cpu")
                    self.assertEqual(evaluated["test"], result["test"])
                    self.assertEqual(evaluated["optimizer_updates"], 0)
                    self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
