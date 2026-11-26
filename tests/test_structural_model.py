"""Behavioral checks for the model update, independent of construction services."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from data_pipeline.preprocessing import load_gog, preprocess_data
from gog_embedding import ComplExDecoder, GoGEncoder
from gog_embedding.relational_ops import relational_aggregate
from gog_embedding.structural import StructuralCategories
from training.checkpoints import initialize_from_checkpoint
from training.experiments import ABLATIONS, run_suite
from training.link_prediction import (EntityOnlyBaseline, TrainingConfig, _backward_in_fact_batches,
                                      _run_one, _training_loss, evaluate, protocol_fingerprints,
                                      sample_training_negatives, selected_facts)


def structural_fixture():
    entities = [{"entity_id": eid, "type": kind, "name": name} for eid, kind, name in [
        ("D1", "drug", "drug one"), ("D2", "drug", "drug two"),
        ("P1", "gene/protein", "protein one"), ("P2", "gene/protein", "protein two"),
        ("X1", "disease", "disease one"), ("X2", "disease", "disease two"),
        ("W1", "pathway", "pathway one"), ("W2", "pathway", "pathway two"),
        ("A", "atom", "C"), ("B", "atom", "O"),
        ("F", "molecular_fragment", "carboxylic_acid"),
        ("S", "molecular_state", "a distinct state"),
        ("R1", "reaction_event", "reaction one"), ("R2", "reaction_event", "reaction two")]]
    graphs = [{"graph_id": "ROOT", "type": "root"},
              *[{"graph_id": gid, "type": "internal", "parent_entity_id": owner}
                for gid, owner in (("DRUG", "D1"), ("PW1", "W1"), ("PW2", "W2"))]]
    nodes = [{"node_id": eid + "@ROOT", "entity_id": eid, "graph_id": "ROOT"}
             for eid in ("D1", "D2", "P1", "P2", "X1", "X2", "W1", "W2")]
    nodes += [{"node_id": eid + "@" + gid, "entity_id": eid, "graph_id": gid}
              for gid, members in (("DRUG", ["A", "B", "F"]), ("PW1", ["S", "R1"]),
                                   ("PW2", ["S", "R2"])) for eid in members]
    relations = []

    def add(h, r, t, split="train", derived=False):
        relations.append(dict(relation_id=f"fact_{len(relations):02}", source_node_id=h,
                              target_node_id=t, type=r, split=split, derived_channel=derived))
    for h, r, t in [("D1", "binds", "P1"), ("D2", "binds", "P2"),
                     ("D1", "treats", "X1"), ("D2", "treats", "X2"),
                     ("P1", "in_pathway", "W1"), ("P2", "in_pathway", "W2")]:
        add(h + "@ROOT", r, t + "@ROOT")
    add("D1@ROOT", "treats", "X2@ROOT", "validation")
    add("D2@ROOT", "treats", "X1@ROOT", "test")
    for h, r, t in [("A@DRUG", "chemical_bond", "B@DRUG"),
                     ("B@DRUG", "chemical_attachment", "F@DRUG"),
                     ("S@PW1", "input_to", "R1@PW1"), ("S@PW2", "input_to", "R2@PW2"),
                     ("P1@ROOT", "same_entity_as", "S@PW1"),
                     ("P2@ROOT", "same_entity_as", "S@PW2"),
                     ("D1@ROOT", "participates_in_reaction", "R1@PW1")]:
        add(h, r, t, derived=True)
    return {"entities": entities, "graphs": graphs, "graph_nodes": nodes, "relations": relations,
            "metadata": {"dataset": "synthetic structural fixture"}}


class StructuralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = preprocess_data(structural_fixture())

    def setUp(self):
        torch.manual_seed(17)

    def model(self, **kwargs):
        return GoGEncoder.from_preprocessed(self.data, hidden_dim=16, num_bases=2,
                                            num_heads=4, dropout=0, profile="structural", **kwargs)

    def test_root_targets_and_negatives_do_not_remove_input_edges(self):
        data = copy.deepcopy(self.data)
        # Exercise the supplied exports' missing flags: ROOT task selection must
        # still keep structural relations out of the prediction objective.
        for fact in data.facts:
            fact.pop("derived_channel", None)
        before = sum(map(len, data.edge_lists.values()))
        self.assertEqual(len(selected_facts(data, "train", "root")), 6)
        negatives, _ = sample_training_negatives(data, 4, 17, "root")
        self.assertTrue(all(i < 6 and data.graph_nodes[n]["graph_id"] == "ROOT" for i, _, n in negatives))
        self.assertEqual(before, sum(map(len, data.edge_lists.values())))
        self.assertEqual(before, 13)

    def test_features_share_categories_without_merging_identities(self):
        module = StructuralCategories(16)
        self.assertEqual(module.categories({"type": "atom", "name": "C"})[1],
                         module.categories({"type": "atom", "name": "C"})[1])
        self.assertEqual(module.categories({"type": "atom", "name": "not-an-element"})[1], 0)
        self.assertEqual(module.categories({"type": "molecular_fragment", "name": "invented"})[3], 0)
        self.assertEqual(module.categories({"type": "molecular_fragment", "name": "ring_system_3"})[4], 3)
        features = module(self.data)
        root_entities = {n["entity_id"] for n in self.data.graph_nodes if n["graph_id"] == "ROOT"}
        self.assertTrue(all(torch.equal(features[self.data.entity_id_to_index[e]], torch.zeros(16))
                            for e in root_entities))
        self.assertFalse(torch.equal(features[self.data.entity_id_to_index["A"]],
                                     features[self.data.entity_id_to_index["B"]]))
        self.assertEqual(sum(n["entity_id"] == "S" for n in self.data.graph_nodes), 2)

    def test_zero_gate_baseline_transfer_is_exact_with_permuted_tables(self):
        baseline = EntityOnlyBaseline(len(self.data.entities), len(self.data.type_id_to_index), 16)
        decoder = ComplExDecoder(len(self.data.relation_id_to_index), 16)
        state = copy.deepcopy(baseline.state_dict())
        dec_state = copy.deepcopy(decoder.state_dict())
        mappings = {"entity_id_to_index": self.data.entity_id_to_index,
                    "type_id_to_index": self.data.type_id_to_index,
                    "relation_id_to_index": self.data.relation_id_to_index}
        checkpoint = {"kind": "baseline", "encoder": state, "decoder": dec_state}
        for key, tensor in (("entity_id_to_index", state["initializer.entity_embedding.weight"]),
                            ("type_id_to_index", state["initializer.type_embedding.weight"]),
                            ("relation_id_to_index", dec_state["relation.weight"])):
            tensor.copy_(tensor.flip(0).clone())
            checkpoint[key] = {label: len(mappings[key]) - 1 - index for label, index in mappings[key].items()}
        model, new_decoder = self.model(), ComplExDecoder(len(self.data.relation_id_to_index), 16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.pt"
            torch.save(checkpoint, path)
            initialize_from_checkpoint(model, new_decoder, self.data, path)
        expected = baseline(self.data)[1]
        actual = model(self.data).z_occurrence
        roots = [i for i, n in enumerate(self.data.graph_nodes) if n["graph_id"] == "ROOT"]
        self.assertTrue(torch.equal(expected[roots], actual[roots]))
        facts = selected_facts(self.data, "validation", "root")
        h = torch.tensor([self.data.node_id_to_index[f["source_node_id"]] for f in facts])
        t = torch.tensor([self.data.node_id_to_index[f["target_node_id"]] for f in facts])
        r = torch.tensor([self.data.relation_id_to_index[f["relation_type_id"]] for f in facts])
        self.assertTrue(torch.equal(decoder.score(expected[h], r, expected[t]),
                                    new_decoder.score(actual[h], r, actual[t])))

    def test_root_loss_reaches_structural_paths_when_gate_is_open(self):
        model, decoder = self.model(), ComplExDecoder(len(self.data.relation_id_to_index), 16)
        with torch.no_grad():
            model.identity_scale.fill_(1)
        output = model(self.data)
        targets = selected_facts(self.data, "train", "root")
        negatives, _ = sample_training_negatives(self.data, 3, 17, "root")
        _backward_in_fact_batches(self.data, decoder, output.z_occurrence, negatives,
                                  targets, self.data.node_id_to_index, self.data.relation_id_to_index, 2)
        for name, parameter in (("local", model.local.basis), ("cross", model.cross.value_node.weight),
                                ("drug_readout", model.structural_readout.heads[0].candidate.weight),
                                ("pathway_readout", model.structural_readout.heads[1].candidate.weight),
                                ("fusion", model.fusion.up_candidate.weight),
                                ("chemistry", model.chemistry.element.weight)):
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_decoder_batching_matches_full_loss_and_encoder_gradient(self):
        model1, decoder1 = self.model(identity_residual=False), ComplExDecoder(len(self.data.relation_id_to_index), 16)
        model2, decoder2 = copy.deepcopy(model1), copy.deepcopy(decoder1)
        targets = selected_facts(self.data, "train", "root")
        negatives, _ = sample_training_negatives(self.data, 2, 17, "root")
        loss = _training_loss(self.data, decoder1, model1(self.data).z_occurrence, negatives,
                              targets, self.data.node_id_to_index, self.data.relation_id_to_index)
        loss.backward()
        batched = _backward_in_fact_batches(self.data, decoder2, model2(self.data).z_occurrence,
                                            negatives, targets, self.data.node_id_to_index,
                                            self.data.relation_id_to_index, 2)
        self.assertAlmostEqual(float(loss.detach()), batched, places=5)
        for (_, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if p1.grad is not None:
                self.assertTrue(torch.allclose(p1.grad, p2.grad, atol=2e-6, rtol=1e-4))

    def test_relation_normalization_preserves_rare_relation_signal(self):
        z = torch.zeros(12, 4)
        z[1] = torch.tensor([1., 2., 3., 4.])
        basis = torch.stack((torch.eye(4)[None], torch.zeros(1, 4, 4)))
        coefficients, relations = torch.ones(2, 2, 1), torch.zeros(2, 4)
        original = relational_aggregate(z, torch.tensor([1]), torch.tensor([0]), torch.tensor([0]),
                                         relations, coefficients, basis, "residual", "relation")
        extended = relational_aggregate(z, torch.arange(1, 12), torch.zeros(11, dtype=torch.long),
                                         torch.tensor([0] + [1] * 10), relations, coefficients,
                                         basis, "residual", "relation")
        self.assertTrue(torch.equal(original[0], extended[0]))
        self.assertGreater(float(extended[0].norm()), 0)
        reversed_edge = relational_aggregate(z, torch.tensor([0]), torch.tensor([1]), torch.tensor([0]),
                                              relations, coefficients, basis, "residual", "relation")
        self.assertTrue(torch.equal(reversed_edge[0], torch.zeros(4)))

    def test_refinement_only_between_rounds_and_shared_identity_only(self):
        model = self.model(num_rounds=2, sync_enabled=True, feedback_enabled=True)
        observed = []
        def capture(module, args, kwargs, output):
            before, after = args[0], output[0]
            count = torch.bincount(args[2], minlength=before.shape[0])
            self.assertTrue(torch.equal(before[count <= 1], after[count <= 1]))
            self.assertTrue(kwargs["active_mask"].any())
            observed.append(output)
        hook = model.sync.register_forward_hook(capture, with_kwargs=True)
        with torch.no_grad():
            model.feedback_scale.zero_()
        result = model(self.data)
        hook.remove()
        self.assertEqual(len(observed), 1)
        self.assertTrue(torch.equal(result.rounds[0].z_root, result.rounds[0].z_next))
        self.assertTrue(torch.equal(result.rounds[-1].z_root, result.rounds[-1].z_next))
        for options in ({"sync_enabled": True}, {"feedback_enabled": True},
                        {"num_rounds": 2, "sync_enabled": True}):
            with self.assertRaises(ValueError):
                self.model(**options)

    def test_no_internal_graphs_empty_heads_and_no_input_mutation(self):
        raw = structural_fixture()
        root_nodes = {n["node_id"] for n in raw["graph_nodes"] if n["graph_id"] == "ROOT"}
        raw["graphs"] = raw["graphs"][:1]
        raw["graph_nodes"] = [n for n in raw["graph_nodes"] if n["node_id"] in root_nodes]
        raw["relations"] = [r for r in raw["relations"] if r["source_node_id"] in root_nodes
                             and r["target_node_id"] in root_nodes]
        data = preprocess_data(raw)
        original = json.dumps(data.to_dict(), sort_keys=True)
        model = GoGEncoder.from_preprocessed(data, hidden_dim=16, num_bases=2, num_heads=4, profile="structural")
        self.assertTrue(torch.isfinite(model(data).z_occurrence).all())
        self.assertEqual(original, json.dumps(data.to_dict(), sort_keys=True))

    def test_legacy_checkpoint_remains_strict_loadable(self):
        model = GoGEncoder.from_preprocessed(self.data, hidden_dim=16, num_bases=2, num_heads=4, dropout=0)
        other = copy.deepcopy(model)
        other.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(model(self.data).z_occurrence, other(self.data).z_occurrence))
        self.assertFalse(any(k.startswith(("identity_scale", "chemistry", "structural_readout", "feedback_scale"))
                             for k in model.state_dict()))

    def test_structural_checkpoint_round_trip_and_starting_checkpoint_eligibility(self):
        config = TrainingConfig(hidden_dim=16, num_bases=2, num_heads=4, dropout=0,
                                negatives_per_side=2, epochs=1, device="cpu", evaluate_test=False)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            baseline = _run_one(self.data, config, "baseline", directory / "baseline")
            from dataclasses import replace
            # An update too small to change float32 weights must preserve the
            # transferred starting checkpoint rather than discard epoch zero.
            updated = _run_one(self.data, replace(config, encoder_profile="structural", task_scope="root",
                                                   learning_rate=1e-30, init_checkpoint=baseline["checkpoint"]),
                                "gog", directory / "structural")
            self.assertEqual(updated["best_epoch"], 0)
            self.assertEqual(updated["initial_validation"], updated["validation"])
            model, decoder = self.model(), ComplExDecoder(len(self.data.relation_id_to_index), 16)
            provenance = initialize_from_checkpoint(model, decoder, self.data, updated["checkpoint"])
            self.assertEqual(provenance["mode"], "exact")
            stored = torch.load(updated["checkpoint"], weights_only=True)
            self.assertTrue(all(torch.equal(v, stored["encoder"][k]) for k, v in model.state_dict().items()))

    def test_root_protocol_ignores_optional_type_restrictions_consistently(self):
        data = copy.deepcopy(self.data)
        data.relation_schema["treats"] = {"head_types": ["drug"], "tail_types": ["disease"]}
        negatives, _ = sample_training_negatives(data, 20, 17, "root")
        targets = selected_facts(data, "train", "root")
        replacements = [data.graph_nodes[n] for i, side, n in negatives
                        if targets[i]["relation_type_id"] == "treats" and side == "head"]
        self.assertTrue(any(data.entities[data.entity_id_to_index[n["entity_id"]]]["type_id"] != "drug"
                            for n in replacements))
        model, decoder = self.model(), ComplExDecoder(len(data.relation_id_to_index), 16)
        report = evaluate(data, decoder, model(data).z_occurrence, "validation", {"train", "validation"}, "root")
        self.assertEqual(report["settings"]["root_candidate_count"], 8)

    def test_supported_ablation_variants_are_finite(self):
        for name, options in ABLATIONS.items():
            with self.subTest(name=name):
                output = self.model(**options)(self.data)
                self.assertTrue(torch.isfinite(output.z_occurrence).all())
                self.assertEqual(len(output.rounds), options.get("num_rounds", 1))

    def test_root_ranking_breakdowns_and_all_candidates(self):
        states = self.model()(self.data).z_occurrence
        decoder = ComplExDecoder(len(self.data.relation_id_to_index), 16)
        report = evaluate(self.data, decoder, states, "validation", {"train", "validation"}, "root")
        self.assertEqual(report["root"]["scorable_queries"], 2)
        self.assertEqual(report["breakdowns"]["by_side"]["head"]["filtered"]["count"], 1)
        self.assertIn("owner_touching", report["breakdowns"]["by_owner"])
        self.assertEqual(report["settings"]["candidates"], "all ROOT occurrences")

    def test_fixed_budget_suite_is_reproducible_and_does_not_evaluate_test(self):
        config = TrainingConfig(hidden_dim=16, num_bases=2, num_heads=4, dropout=0,
                                negatives_per_side=2, training_batch_size=4,
                                device="cpu", evaluate_test=False, diagnostic_train_limit=2)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_suite(self.data, first, config, seeds=[17], warmup_updates=2, continuation_updates=2)
            b = run_suite(self.data, second, config, seeds=[17], warmup_updates=2, continuation_updates=2)
            self.assertEqual(a["manifest"], b["manifest"])
            self.assertEqual(a["aggregate"], b["aggregate"])
            for label, result in a["runs"][0]["models"].items():
                self.assertIsNone(result["test"])
                self.assertEqual(result["training_target_count"], 6)
                self.assertEqual(result["optimizer_updates"], 2 if label in
                                 {"baseline_continuation", "gog_structural"} else 4)
                self.assertEqual(result["total_budget_updates"], 4)
                self.assertEqual(result["fingerprints"], protocol_fingerprints(self.data, "root"))
                checkpoint = torch.load(result["checkpoint"], weights_only=True)
                self.assertIn("type_id_to_index", checkpoint)
                if label == "gog_structural":
                    self.assertIn("identity_scale", checkpoint["encoder"])
                    self.assertIsNotNone(checkpoint["feature_vocabularies"])
            self.assertTrue((Path(first) / "comparison.csv").exists())
            baseline = a["runs"][0]["models"]["baseline_continuation"]["initial_validation"]
            structural = a["runs"][0]["models"]["gog_structural"]["initial_validation"]
            self.assertEqual(baseline, structural)


if __name__ == "__main__":
    unittest.main()
