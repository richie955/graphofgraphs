import unittest
from pathlib import Path

import torch

from gog_model import (GoGEncoder, GoGEncoderStages1To3, Stage4GraphReadout,
                       Stage5GraphEntityFusion, Stage6RootCompGCN,
                       Stage7IdentitySync, Stage8ContextFeedback)
from gog_model.stage1 import Stage1Initializer
from gog_model.stage3 import Stage3CrossGraphAttention
from gog_preprocessing import load_gog


FIXTURE = Path(__file__).parent / "docs" / "synthetic_gog_fixture.json"


class Stage1Tests(unittest.TestCase):
    def test_shapes_shared_identity_unknown_type_and_empty_occurrences(self):
        module = Stage1Initializer(num_entities=3, num_types=2, hidden_dim=8)
        types = torch.tensor([0, 1, 1])
        occurrence_entities = torch.tensor([0, 1, 0])
        b, delta, z = module(types, occurrence_entities)
        self.assertEqual(tuple(b.shape), (3, 8))
        self.assertEqual(tuple(delta.shape), (3, 8))
        self.assertEqual(tuple(z.shape), (3, 8))
        self.assertTrue(torch.equal(z[0], z[2]))
        self.assertTrue(torch.isfinite(z).all())
        b0, d0, z0 = module(types, torch.empty(0, dtype=torch.long))
        self.assertEqual(tuple(d0.shape), (0, 8))
        self.assertEqual(tuple(z0.shape), (0, 8))
        self.assertEqual(tuple(b0.shape), (3, 8))

    def test_gradients_and_no_occurrence_parameter(self):
        module = Stage1Initializer(num_entities=4, num_types=3, hidden_dim=8)
        b, _, z = module(torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 0, 3]))
        (b.square().mean() + z.square().mean()).backward()
        self.assertIsNotNone(module.entity_embedding.weight.grad)
        self.assertIsNotNone(module.type_embedding.weight.grad)
        self.assertIsNotNone(module.occurrence_norm.weight.grad)
        self.assertFalse(any("occurrence" in name and "norm" not in name for name, _ in module.named_parameters()))


class Stage3Tests(unittest.TestCase):
    def test_scope_for_both_directions(self):
        root = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([1, 1, 0, 1])
        scopes = Stage3CrossGraphAttention._channel_scope(root, target, 0)
        self.assertEqual(scopes.tolist(), [0, 0, 1, 2])

    def test_segmented_softmax_normalizes_per_receiver_and_head(self):
        scores = torch.tensor([[1.0, 2.0], [3.0, 1.0], [4.0, 6.0]])
        receivers = torch.tensor([0, 0, 1])
        weights = Stage3CrossGraphAttention._segmented_softmax(scores, receivers, 2)
        self.assertTrue(torch.allclose(weights[:2].sum(0), torch.ones(2)))
        self.assertTrue(torch.allclose(weights[2], torch.ones(2)))

    def test_empty_edges_preserve_input_exactly(self):
        module = Stage3CrossGraphAttention(num_relations=2, hidden_dim=8, num_heads=2, dropout=0.0)
        z = torch.randn(3, 8)
        out, count, scopes = module(z, torch.empty((2, 0), dtype=torch.long),
                                    torch.empty((0,), dtype=torch.long), torch.randn(2, 8),
                                    torch.tensor([0, 1, 2]), root_graph_index=0)
        self.assertTrue(torch.equal(z, out))
        self.assertEqual(count, 0)
        self.assertEqual(scopes.numel(), 0)


class IntegrationTests(unittest.TestCase):
    def test_fixture_full_slice_and_gradients(self):
        data = load_gog(FIXTURE)
        model = GoGEncoderStages1To3.from_preprocessed(data, hidden_dim=32, num_bases=2, num_heads=4, dropout=0.0)
        output = model(data)
        n_entity, n_occurrence = len(data.entities), len(data.graph_nodes)
        for tensor in (output.b_entity, output.delta_occurrence, output.z_occurrence,
                       output.z_local, output.z_cross):
            self.assertEqual(tuple(tensor.shape), (n_entity if tensor is output.b_entity else n_occurrence, 32))
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(output.local_channel_count, 10)  # five train facts, two directions
        self.assertEqual(output.cross_channel_count, 12)  # six train facts, two directions
        root_index = data.graph_id_to_index["G_ROOT"]
        root_occurrences = [i for i, graph_index in enumerate(data.graph_index_by_occurrence) if graph_index == root_index]
        self.assertTrue(torch.equal(output.z_local[root_occurrences], output.z_occurrence[root_occurrences]))
        internal_occurrences = [i for i, graph_index in enumerate(data.graph_index_by_occurrence) if graph_index != root_index]
        self.assertFalse(torch.equal(output.z_local[internal_occurrences], output.z_occurrence[internal_occurrences]))
        held_out = {"F003", "F007"}
        routed = [edge.fact_id for edges in data.edge_lists.values() for edge in edges]
        self.assertFalse(held_out.intersection(routed))
        self.assertEqual(len(routed), len(set(routed)))
        output.z_cross.square().mean().backward()
        self.assertIsNotNone(model.initializer.entity_embedding.weight.grad)
        self.assertIsNotNone(model.relation_embedding.weight.grad)
        self.assertIsNotNone(model.local.basis.grad)
        self.assertIsNotNone(model.cross.query.weight.grad)

    def test_fixture_type_mapping_includes_unknown(self):
        data = load_gog(FIXTURE)
        self.assertIn("UNK", data.type_id_to_index)
        self.assertEqual(len(data.type_index_by_entity), len(data.entities))
        self.assertEqual(data.type_id_to_index, {"Disease": 0, "Drug": 1, "Pathway": 2, "Process": 3, "Protein": 4, "UNK": 5})

    def test_two_round_full_encoder_and_gradients(self):
        data = load_gog(FIXTURE)
        model = GoGEncoder.from_preprocessed(data, hidden_dim=32, num_bases=2, num_heads=4,
                                             dropout=0.0, num_rounds=2)
        output = model(data)
        self.assertEqual(len(output.rounds), 2)
        self.assertEqual(tuple(output.b_entity.shape), (13, 32))
        self.assertEqual(tuple(output.z_occurrence.shape), (14, 32))
        self.assertEqual(tuple(output.h_graph.shape), (2, 32))
        for state in (output.b_entity, output.z_occurrence, output.h_graph):
            self.assertTrue(torch.isfinite(state).all())
        self.assertEqual([r.root_channel_count for r in output.rounds], [4, 4])
        (output.b_entity.square().mean() + output.z_occurrence.square().mean() +
         output.h_graph.square().mean()).backward()
        for parameter in (model.readout.attention_vector, model.fusion.up_gate.weight,
                          model.root.basis, model.sync.score_vector, model.feedback.gate.weight):
            self.assertIsNotNone(parameter.grad)


class Stages4To8Tests(unittest.TestCase):
    def test_stage4_singleton_and_empty_graph(self):
        module = Stage4GraphReadout(hidden_dim=8, dropout=0.0)
        z = torch.randn(1, 8)
        previous = torch.randn(2, 8)
        updated, attention, pooled = module(z, torch.tensor([0]), torch.tensor([0]), previous)
        self.assertEqual(attention.tolist(), [1.0])
        self.assertTrue(torch.equal(pooled[0], z[0]))
        self.assertTrue(torch.equal(updated[1], previous[1]))

    def test_stage5_only_updates_nonempty_owner(self):
        module = Stage5GraphEntityFusion(hidden_dim=8, dropout=0.0)
        z = torch.randn(3, 8)
        graphs = torch.randn(2, 8)
        z_out, graph_out, feedback = module(z, graphs, torch.tensor([0, 1]), torch.tensor([True, False]))
        self.assertFalse(torch.equal(z_out[0], z[0]))
        self.assertTrue(torch.equal(z_out[1:], z[1:]))
        self.assertTrue(torch.equal(graph_out[1], graphs[1]))
        self.assertTrue(torch.equal(feedback[1], torch.zeros(8)))

    def test_stage6_updates_root_only_even_without_edges(self):
        module = Stage6RootCompGCN(num_relations=2, hidden_dim=8, num_bases=2, dropout=0.0)
        z = torch.randn(3, 8)
        rel = torch.randn(2, 8)
        out, count = module(z, torch.tensor([0, 1]), torch.tensor([[0], [1]]), torch.tensor([0]), rel)
        self.assertEqual(count, 2)
        self.assertTrue(torch.equal(out[2], z[2]))
        no_edge, no_count = module(z, torch.tensor([0, 1]), torch.empty((2, 0), dtype=torch.long),
                                   torch.empty((0,), dtype=torch.long), rel)
        self.assertEqual(no_count, 0)
        self.assertFalse(torch.equal(no_edge[:2], z[:2]))
        self.assertTrue(torch.equal(no_edge[2], z[2]))

    def test_stage7_single_occurrence_and_absent_entity(self):
        module = Stage7IdentitySync(hidden_dim=8)
        b = torch.randn(3, 8)
        z = torch.randn(2, 8)
        out, attention = module(b, z, torch.tensor([0, 1]), torch.tensor([True, False]))
        self.assertEqual(attention.tolist(), [1.0, 0.0])
        self.assertFalse(torch.equal(out[0], b[0]))
        self.assertTrue(torch.equal(out[1:], b[1:]))

    def test_stage8_rebuilds_occurrences_from_shared_entities(self):
        module = Stage8ContextFeedback(hidden_dim=8, scope_dim=2, dropout=0.0)
        b = torch.randn(2, 8)
        z = torch.randn(3, 8)
        h = torch.randn(1, 8)
        z_next, delta = module(
            b, z, h,
            entity_index_by_occurrence=torch.tensor([0, 0, 1]),
            graph_index_by_occurrence=torch.tensor([0, 1, 1]),
            graph_to_internal_index=torch.tensor([-1, 0]),
            root_occurrence_by_entity=torch.tensor([0, -1]),
            owned_graph_by_entity=torch.tensor([-1, -1]),
            has_owned_graph=torch.tensor([False, False]),
            owner_root_by_internal_graph=torch.tensor([0]),
            has_graph_owner_root=torch.tensor([True]),
            graph_nonempty=torch.tensor([True]), root_graph_index=0)
        self.assertEqual(tuple(z_next.shape), (3, 8))
        self.assertEqual(tuple(delta.shape), (3, 8))
        self.assertTrue(torch.isfinite(z_next).all())
        self.assertFalse(torch.equal(z_next[0], z_next[1]))


if __name__ == "__main__":
    unittest.main()
