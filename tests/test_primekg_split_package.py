"""End-to-end checks for the source-fact split artifact."""

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from gog_data.primekg_split_package import SplitParameters, build_split_package


class PrimeKGSplitPackageTests(unittest.TestCase):
    def test_reproducible_pair_safe_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "kg.csv"
            entities = [(f"E{i:02d}", ("drug", "gene/protein", "pathway", "disease")[i % 4])
                        for i in range(36)]
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["x_index", "x_type", "x_name", "y_index",
                                                            "y_type", "y_name", "relation", "display_relation"])
                writer.writeheader()
                for i, (left, left_type) in enumerate(entities):
                    for j, (right, right_type) in enumerate(entities[i + 1:], i + 1):
                        relation = ("associated", "interacts", "linked")[((i + j) % 3)]
                        row = {"x_index": left, "x_type": left_type, "x_name": left,
                               "y_index": right, "y_type": right_type, "y_name": right,
                               "relation": relation, "display_relation": relation}
                        writer.writerow(row)
                        if (i + j) % 11 == 0:
                            writer.writerow(row)  # an exact source duplicate
                        if (i + j) % 7 == 0:
                            writer.writerow({**row, "x_index": right, "x_type": right_type,
                                             "x_name": right, "y_index": left,
                                             "y_type": left_type, "y_name": left})
            params = SplitParameters(target_facts=120, owners_per_type=2, min_entities=20,
                                     max_entities=36, min_owner_train_groups=2)
            first, second = root / "first.zip", root / "second.zip"
            report1 = build_split_package(source, first, params)
            report2 = build_split_package(source, second, params)
            self.assertEqual(report1["selected_fact_set_sha256"], report2["selected_fact_set_sha256"])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = set(archive.namelist())
                self.assertTrue({"selected_facts.csv", "split_manifest.csv", "selected_entities.csv",
                                 "owners.csv", "parameters.json", "audit_report.json",
                                 "duplicate_source_rows.csv"}.issubset(names))
                facts = pd.read_csv(io.BytesIO(archive.read("selected_facts.csv")))
                manifest = pd.read_csv(io.BytesIO(archive.read("split_manifest.csv")))
                audit = json.loads(archive.read("audit_report.json"))
                owners = pd.read_csv(io.BytesIO(archive.read("owners.csv")))
            self.assertEqual(set(facts["fact_id"]), set(manifest["fact_id"]))
            self.assertEqual(manifest.groupby("pair_key")["split"].nunique().max(), 1)
            self.assertEqual(audit["validation"], "passed")
            self.assertEqual(set(facts["split"]), {"train", "validation", "test"})
            self.assertEqual(owners["type"].value_counts().to_dict(),
                             {"drug": 2, "gene/protein": 2, "pathway": 2})
            self.assertGreater(audit["source"]["exact_duplicate_rows"], 0)


if __name__ == "__main__":
    unittest.main()
