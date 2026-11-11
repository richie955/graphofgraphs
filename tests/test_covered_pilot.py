"""Fixed-source-split pilot audits and one-epoch integration test."""

import tempfile
import unittest
from pathlib import Path

from training.covered_pilot import (FlattenedContextRGCN, _negative_options,
                                    load_covered_experiment, run_covered_pilot)
from training.link_prediction import TrainingConfig


BASE = Path(__file__).parent.parent
GOG = BASE / "prime_kg_gog_from_training.json"
TRAIN = BASE / "facts/train_data.json"
VALIDATION = BASE / "facts/validation_data.json"
TEST = BASE / "facts/test_data.json"


@unittest.skipUnless(all(p.is_file() for p in (GOG, TRAIN, VALIDATION, TEST)),
                     "current PrimeKG pilot files are unavailable")
class CoveredPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = load_covered_experiment(GOG, TRAIN, VALIDATION, TEST)

    def test_fixed_source_splits_are_separate_from_training_context(self):
        x = self.experiment
        self.assertEqual(x.audit["covered_counts"], {"train": 2936, "validation": 226, "test": 256})
        self.assertEqual(x.audit["covered_pair_counts"]["test"], 128)
        self.assertEqual(len(x.gog.edge_lists["root_to_root"]), 1468)
        self.assertEqual(sum(len(edges) for edges in x.gog.edge_lists.values()), 3345)
        self.assertFalse(x.gog.report["splits_generated"])
        self.assertTrue(all(fact["split"] == "train" for fact in x.gog.facts))
        pairs = {split: {fact.pair for fact in rows} for split, rows in x.facts.items()}
        self.assertFalse(pairs["train"] & pairs["validation"])
        self.assertFalse(pairs["train"] & pairs["test"])
        self.assertFalse(pairs["validation"] & pairs["test"])

    def test_training_negatives_are_type_compatible_and_exclude_train_truths(self):
        x = self.experiment
        options = _negative_options(x)
        known = {(f.head, f.relation, f.tail) for f in x.facts["train"]}
        self.assertEqual(len(options), 2936)
        for fact, sides in zip(x.facts["train"], options):
            self.assertTrue(set(sides["head"]).issubset(x.typed_candidates[(fact.relation, "head")]))
            self.assertTrue(set(sides["tail"]).issubset(x.typed_candidates[(fact.relation, "tail")]))
            self.assertTrue(all((candidate, fact.relation, fact.tail) not in known
                                for candidate in sides["head"]))
            self.assertTrue(all((fact.head, fact.relation, candidate) not in known
                                for candidate in sides["tail"]))

    def test_flattened_control_receives_semantic_and_structural_context(self):
        data = self.experiment.gog
        model = FlattenedContextRGCN(data, hidden_dim=16, num_bases=2, dropout=0)
        internal_occurrences = sum(node["graph_id"] != "ROOT" for node in data.graph_nodes)
        internal_graphs = sum(graph["kind"] == "internal" for graph in data.graphs)
        expected = 2 * (3345 + internal_occurrences * 2 + internal_graphs)
        self.assertEqual(model.edge_index.shape, (2, expected))
        self.assertEqual(set(model.edge_type.tolist()),
                         set(range(2 * len(data.relation_id_to_index) + 6)))

    def test_one_epoch_all_models_share_evaluation_questions_and_candidates(self):
        config = TrainingConfig(hidden_dim=16, num_bases=2, num_heads=4, dropout=0,
                                negatives_per_side=1, training_batch_size=512,
                                epochs=1, patience=1, seed=17, device="cpu")
        with tempfile.TemporaryDirectory() as temporary:
            report = run_covered_pilot(GOG, TRAIN, VALIDATION, TEST, temporary,
                                       config, seeds=(17,))
            self.assertEqual(len(report["paired_comparisons"]), 2)
            self.assertTrue((Path(temporary) / "covered_pilot_report.json").is_file())
            model_runs = [report["models"][name][0] for name in
                          ("flat_root", "flat_context", "gog")]
            self.assertTrue(all(run["test_typed"]["facts"] == 256 for run in model_runs))
            self.assertTrue(all(run["test_typed"]["pairs"] == 128 for run in model_runs))
            self.assertTrue(all(run["test_all_root"]["queries"] == 512 for run in model_runs))
            import json
            details = [json.loads((Path(temporary) / run["report"]).read_text(encoding="utf-8"))
                       for run in model_runs]
            for mode in ("typed", "all_root"):
                signatures = [[(q["fact_id"], q["side"], q["candidates"])
                               for q in run["test"][mode]["queries"]] for run in details]
                self.assertEqual(signatures[0], signatures[1])
                self.assertEqual(signatures[0], signatures[2])


if __name__ == "__main__":
    unittest.main()
