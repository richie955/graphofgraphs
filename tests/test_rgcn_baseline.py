import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from baselines.rgcn import RGCNBaseline, training_channels
from data_pipeline.preprocessing import load_gog
from training.link_prediction import TrainingConfig, run_rgcn


FIXTURE = Path(__file__).parent.parent / "docs" / "synthetic_gog_fixture.json"


class RGCNBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_gog(FIXTURE)

    def test_channels_use_each_training_fact_once_in_each_direction(self):
        data = self.data
        edge_index, edge_type = training_channels(data, torch.device("cpu"))
        train_edges = [edge for edges in data.edge_lists.values() for edge in edges]
        count = len(train_edges)
        relation_count = len(data.relation_id_to_index)
        self.assertEqual(tuple(edge_index.shape), (2, 2 * count))
        self.assertEqual(tuple(edge_type.shape), (2 * count,))
        forward = Counter((int(edge_index[0, i]), int(edge_type[i]), int(edge_index[1, i]))
                          for i in range(count))
        reverse = Counter((int(edge_index[0, i]), int(edge_type[i]), int(edge_index[1, i]))
                          for i in range(count, 2 * count))
        expected = Counter((edge.source_occurrence_index, edge.relation_index,
                            edge.target_occurrence_index) for edge in train_edges)
        self.assertEqual(forward, expected)
        self.assertEqual(reverse, Counter({(target, relation + relation_count, source): count
                                           for (source, relation, target), count in expected.items()}))
        self.assertEqual(count, sum(len(edges) for edges in data.edge_lists.values()))
        self.assertEqual({edge.fact_id for edge in train_edges} & set(data.report["held_out_fact_ids"]), set())

    def test_forward_and_gradients(self):
        model = RGCNBaseline.from_preprocessed(self.data, hidden_dim=16, num_bases=2, dropout=0)
        b_entity, z_occurrence = model(self.data)
        self.assertEqual(tuple(b_entity.shape), (len(self.data.entities), 16))
        self.assertEqual(tuple(z_occurrence.shape), (len(self.data.graph_nodes), 16))
        self.assertTrue(torch.isfinite(z_occurrence).all())
        z_occurrence.square().mean().backward()
        for parameter in (model.initializer.entity_embedding.weight,
                          model.conv1.weight, model.conv2.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_short_training_run(self):
        config = TrainingConfig(hidden_dim=16, num_bases=2, dropout=0,
                                negatives_per_side=2, training_batch_size=4,
                                epochs=2, patience=2, seed=23, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            report = run_rgcn(self.data, config, directory)
            self.assertEqual(report["rgcn"]["kind"], "rgcn")
            self.assertTrue((Path(directory) / "rgcn_best.pt").exists())
            self.assertTrue((Path(directory) / "rgcn_report.json").exists())
            self.assertIn("filtered", report["rgcn"]["test"]["aggregate"])


if __name__ == "__main__":
    unittest.main()
