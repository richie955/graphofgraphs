import tempfile
import unittest
from pathlib import Path

import torch

from gog_model.stage9 import ComplExDecoder
from gog_preprocessing import load_gog
from stage9_training import TrainingConfig, known_fact_keys, run_comparison, sample_training_negatives, valid_candidate_nodes


FIXTURE = Path(__file__).parent / "docs" / "synthetic_gog_fixture.json"


class Stage9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_gog(FIXTURE)

    def test_decoder_scores_directed_triples_and_receives_gradients(self):
        decoder = ComplExDecoder(num_relations=len(self.data.relation_id_to_index), hidden_dim=16)
        head, tail = torch.randn(2, 16), torch.randn(2, 16)
        rel = torch.tensor([0, 1])
        forward = decoder.score(head, rel, tail)
        reverse = decoder.score(tail, rel, head)
        self.assertFalse(torch.allclose(forward, reverse))
        forward.sum().backward()
        self.assertIsNotNone(decoder.projection.weight.grad)
        self.assertIsNotNone(decoder.relation.weight.grad)

    def test_negative_candidates_match_type_and_context_and_exclude_training_truths(self):
        train_fact = next(f for f in self.data.facts if f["split"] == "train")
        known = known_fact_keys(self.data, {"train"})
        nodes = {n["node_id"]: n for n in self.data.graph_nodes}
        entities = {e["entity_id"]: e for e in self.data.entities}
        for side in ("head", "tail"):
            candidates = valid_candidate_nodes(self.data, train_fact, side, known)
            for index in candidates:
                node = self.data.graph_nodes[index]
                self.assertEqual(node["graph_id"], nodes[train_fact["source_node_id"] if side == "head" else train_fact["target_node_id"]]["graph_id"])
                relation = self.data.relation_schema[train_fact["relation_type_id"]]
                allowed = relation["head_types" if side == "head" else "tail_types"]
                self.assertIn(entities[node["entity_id"]]["type_id"], allowed)
        sampled, _ = sample_training_negatives(self.data, negatives_per_side=3, seed=5)
        train_facts = [f for f in self.data.facts if f["split"] == "train"]
        for fact_index, side, candidate_idx in sampled:
            fact = train_facts[fact_index]
            candidate_id = self.data.graph_nodes[candidate_idx]["node_id"]
            self.assertNotIn("F003", {f["fact_id"] for f in train_facts})
            self.assertNotIn("F007", {f["fact_id"] for f in train_facts})
            source_graph = nodes[fact["source_node_id"]]["graph_id"]
            target_graph = nodes[fact["target_node_id"]]["graph_id"]
            key = ((candidate_id if side == "head" else fact["source_node_id"]), fact["relation_type_id"],
                   (fact["target_node_id"] if side == "head" else candidate_id), source_graph, target_graph)
            self.assertNotIn(key, known)

    def test_fixture_training_comparison_writes_labeled_report(self):
        config = TrainingConfig(hidden_dim=16, num_bases=2, num_heads=4, dropout=0.0,
                                negatives_per_side=2, epochs=5, patience=3,
                                learning_rate=0.01, seed=23, device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            report = run_comparison(self.data, config, temp)
            self.assertIn("not evidence", report["notice"])
            self.assertEqual(set(report["training_facts"]), {f["fact_id"] for f in self.data.facts if f["split"] == "train"})
            self.assertEqual(set(report["validation_facts"]), {"F003"})
            self.assertEqual(set(report["test_facts"]), {"F007"})
            self.assertTrue((Path(temp) / "baseline_best.pt").exists())
            self.assertTrue((Path(temp) / "gog_best.pt").exists())
            self.assertTrue((Path(temp) / "comparison_report.json").exists())
            for kind in ("baseline", "gog"):
                self.assertGreaterEqual(report[kind]["best_epoch"], 1)
                self.assertGreaterEqual(report[kind]["validation"]["root"]["scorable_queries"], 0)
                self.assertIn("filtered", report[kind]["test"]["aggregate"])


if __name__ == "__main__":
    unittest.main()
