import copy
import json
import unittest
from pathlib import Path

from gog_preprocessing import GoGValidationError, load_gog, preprocess_data


FIXTURE = Path(__file__).parent / "docs" / "synthetic_gog_fixture.json"


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.result = preprocess_data(cls.raw)

    def test_fixture_counts_and_expected_scope_counts(self):
        result = self.result
        self.assertEqual(result.report["counts"], {
            "entities": 13, "entity_types": 5, "graphs": 3,
            "internal_graphs": 2, "graph_nodes": 14, "facts": 15,
        })
        self.assertEqual(result.report["scope_counts_by_split"]["train"], {
            "root_to_root": 2, "same_internal_graph": 5, "root_to_internal": 3,
            "internal_to_root": 1, "internal_to_different_internal": 2,
        })

    def test_only_training_facts_routed_and_each_once(self):
        routed = [edge for edges in self.result.edge_lists.values() for edge in edges]
        self.assertEqual(len(routed), 13)
        self.assertEqual(len({edge.fact_id for edge in routed}), 13)
        self.assertTrue(all(edge.split == "train" and edge.direction == "FORWARD" for edge in routed))
        self.assertFalse({"F003", "F007"} & {edge.fact_id for edge in routed})
        self.assertEqual(set(self.result.report["held_out_fact_ids"]), {"F003", "F007"})

    def test_scope_routing_and_reverse_is_not_generated(self):
        for scope, edges in self.result.edge_lists.items():
            self.assertTrue(all(edge.scope == scope for edge in edges))
        self.assertEqual(self.result.report["reverse_channels_generated"], False)
        self.assertEqual(self.result.report["training_edge_counts_by_scope"], {
            "root_to_root": 2, "same_internal_graph": 5, "root_to_internal": 3,
            "internal_to_root": 1, "internal_to_different_internal": 2,
        })

    def test_owner_and_root_maps(self):
        self.assertEqual(len(self.result.owner_entity_by_graph), 2)
        self.assertTrue(all(self.result.has_owner_root))
        mek = self.result.entity_id_to_index["E_MEK"]
        self.assertTrue(self.result.has_root_occurrence[mek])
        self.assertGreaterEqual(self.result.root_occurrence_by_entity[mek], 0)

    def test_mappings_are_deterministic(self):
        second = load_gog(FIXTURE)
        self.assertEqual(self.result.entity_id_to_index, second.entity_id_to_index)
        self.assertEqual(self.result.node_id_to_index, second.node_id_to_index)
        self.assertEqual(self.result.to_dict()["edge_lists"], second.to_dict()["edge_lists"])

    def assert_invalid(self, mutate, expected):
        data = copy.deepcopy(self.raw)
        mutate(data)
        with self.assertRaisesRegex(GoGValidationError, expected):
            preprocess_data(data)

    def test_duplicate_entity_id_rejected(self):
        self.assert_invalid(lambda d: d["entities"].append(copy.deepcopy(d["entities"][0])), "duplicate entity IDs")

    def test_duplicate_occurrence_pair_rejected(self):
        self.assert_invalid(lambda d: d["graph_nodes"].append({"node_id": "N_DUP", "entity_id": "E_MEK", "graph_id": "G_MAPK"}), "duplicate entity/graph occurrence pair")

    def test_missing_relation_endpoint_rejected(self):
        self.assert_invalid(lambda d: d["facts"][0].update(source_node_id="NO_SUCH_NODE"), "unresolved endpoint")

    def test_invalid_owner_rejected(self):
        self.assert_invalid(lambda d: d["graphs"][1].update(owner_entity_id="NO_SUCH_ENTITY"), "unknown or missing owner")

    def test_owner_without_root_occurrence_rejected(self):
        self.assert_invalid(lambda d: d["graph_nodes"].remove(next(n for n in d["graph_nodes"] if n["node_id"] == "N_MAPK_ROOT")), "has no ROOT GraphNode")

    def test_wrong_endpoint_type_rejected(self):
        self.assert_invalid(lambda d: d["facts"][0].update(relation_type_id="activates"), "head type .* invalid")

    def test_multiple_root_graphs_rejected(self):
        self.assert_invalid(lambda d: d["graphs"].append({"graph_id": "G_ROOT2", "kind": "root", "owner_entity_id": None}), "expected exactly one ROOT")

    def test_invalid_split_rejected(self):
        self.assert_invalid(lambda d: d["facts"][0].update(split="dev"), "invalid split")


if __name__ == "__main__":
    unittest.main()
