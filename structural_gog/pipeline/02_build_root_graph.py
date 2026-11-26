#!/usr/bin/env python3
"""Phase 1: build the root graph from the seed-and-neighbourhood subset rule.

Seeds = drugs that resolved to a usable SMILES + pathways under the node limit,
taken from data/interim/ (run 01_coverage_check.py first). Subset = their k-hop
neighbourhood, largest connected component.

Writes data/out/root_graph.json and data/interim/seeds.json.

    python scripts/02_build_root_graph.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog import primekg
from gog.config import cfg
from gog.schema import Edge, Node, root_id


def main() -> int:
    conf = cfg()
    interim, out = conf["paths"]["interim_dir"], conf["paths"]["out_dir"]

    struct_path = interim / "drug_structures.json"
    sizes_path = interim / "pathway_sizes.json"
    if not struct_path.exists() or not sizes_path.exists():
        print("run scripts/01_coverage_check.py first", file=sys.stderr)
        return 1

    max_atoms = conf["subset"]["max_heavy_atoms"]
    max_nodes = conf["subset"]["max_pathway_nodes"]

    min_nodes = conf["subset"].get("min_pathway_nodes", 1)
    structures = json.loads(struct_path.read_text())
    seed_drugs = [
        db for db, r in structures.items()
        if r.get("smiles") and (r.get("heavy_atoms") is None or r["heavy_atoms"] <= max_atoms)
    ]
    sizes = json.loads(sizes_path.read_text())
    seed_pathways = [s["st_id"] for s in sizes if min_nodes <= s["n_nodes"] <= max_nodes]
    print(f"seeds: {len(seed_drugs)} drugs, {len(seed_pathways)} pathways")
    if not seed_drugs:
        print("no usable drug seeds; re-run the coverage check", file=sys.stderr)
        return 1

    df = primekg.load_primekg()
    print(f"full PrimeKG: {len(df):,} edges")

    seed_keys = {f"drug:{d}" for d in seed_drugs} | {f"pathway:{p}" for p in seed_pathways}
    hub = set(conf["subset"].get("hub_relations") or [])
    edges_df = primekg.expand_subset(
        df, seed_keys, hops=conf["subset"]["root_hops"], exclude_relations=hub
    )
    print(f"after {conf['subset']['root_hops']}-hop expansion: {len(edges_df):,} edges")
    if conf["subset"]["keep_largest_component"]:
        edges_df = primekg.largest_component(edges_df)
        print(f"largest component: {len(edges_df):,} edges")

    node_rows = primekg.node_table(edges_df)
    print(f"subset: {len(node_rows):,} nodes, {len(edges_df):,} edges")

    nodes = [
        Node(
            id=root_id(r.type, r.id),
            type=r.type,
            name=str(r.name),
            attrs={"source": str(r.source), "source_id": r.id},
        )
        for r in node_rows.itertuples()
    ]
    edges = [
        Edge(
            source=root_id(r.x_type, r.x_id),
            target=root_id(r.y_type, r.y_id),
            type=r.relation,
            attrs={"display_relation": r.display_relation},
        )
        for r in edges_df.itertuples()
    ]

    type_counts = Counter(n.type for n in nodes)
    rel_counts = Counter(e.type for e in edges)

    from dataclasses import asdict

    doc = {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "metadata": {
            "subset_rule": {
                "seed_drugs": len(seed_drugs),
                "seed_pathways": len(seed_pathways),
                "hops": conf["subset"]["root_hops"],
                "largest_component": conf["subset"]["keep_largest_component"],
                "max_heavy_atoms": max_atoms,
                "max_pathway_nodes": max_nodes,
            },
            "node_type_counts": dict(type_counts),
            "relation_counts": dict(rel_counts),
        },
    }
    (out / "root_graph.json").write_text(json.dumps(doc, indent=2))
    (interim / "seeds.json").write_text(
        json.dumps({"drugs": sorted(seed_drugs), "pathways": sorted(seed_pathways)}, indent=2)
    )

    print("\nnode types:", dict(type_counts.most_common()))
    print(f"\nwrote {out / 'root_graph.json'}")
    print("Owner nodes for Phases 2 and 3 are in", interim / "seeds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
