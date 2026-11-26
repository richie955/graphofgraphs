#!/usr/bin/env python3
"""Phase 2: build a fragment-level internal graph for every seed drug.

Reads data/interim/drug_structures.json (written by the coverage check) and the
seed list, and writes data/out/drug_internal_graphs.json. Needs no network.

Every molecule is validated before it is accepted: the atom partition must be
complete and the original bond set must be recoverable from the fragments. A
molecule that fails is reported and skipped, never silently half-written.

    python scripts/04_build_drug_graphs.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from gog.config import cfg
from gog.fragments import build_drug_graph


def main() -> int:
    conf = cfg()
    interim, out = conf["paths"]["interim_dir"], conf["paths"]["out_dir"]
    max_atoms = conf["subset"]["max_heavy_atoms"]

    struct_path = interim / "drug_structures.json"
    seeds_path = interim / "seeds.json"
    if not struct_path.exists():
        print("run scripts/01_coverage_check.py first", file=sys.stderr)
        return 1
    structures = json.loads(struct_path.read_text())

    if seeds_path.exists():
        wanted = set(json.loads(seeds_path.read_text()).get("drugs", []))
        print(f"restricting to {len(wanted)} seed drugs from the root graph")
    else:
        wanted = set(structures)
        print("no seeds.json; building every resolved drug")

    todo = [(db, rec) for db, rec in structures.items()
            if db in wanted and rec.get("smiles")]
    print(f"building {len(todo)} drug internal graphs (limit {max_atoms} heavy atoms)")

    graphs = []
    reports = []
    status = Counter()
    frag_labels: Counter[str] = Counter()
    for db, rec in tqdm(todo, unit="drug"):
        g, r = build_drug_graph(db, rec["smiles"], pubchem_cid=rec.get("cid"),
                                max_heavy_atoms=max_atoms)
        reports.append(r)
        status[r["status"]] += 1
        if g is not None:
            graphs.append(g)
            for n in g.nodes:
                if n.type == "molecular_fragment":
                    frag_labels[n.attrs["group_name"]] += 1

    payload = {
        "metadata": {
            "n_graphs": len(graphs),
            "status_counts": dict(status),
            "max_heavy_atoms": max_atoms,
            "fragmenter": "ring_systems+smarts_functional_groups/v1",
            "fragment_label_counts": dict(frag_labels.most_common()),
        },
        "graphs": [asdict(g) for g in graphs],
    }
    (out / "drug_internal_graphs.json").write_text(json.dumps(payload, indent=2))
    (interim / "drug_graph_reports.json").write_text(json.dumps(reports, indent=2))

    n_nodes = sum(len(g.nodes) for g in graphs)
    n_edges = sum(len(g.edges) for g in graphs)
    print("\n" + "=" * 64)
    print(f"built     {len(graphs):,} graphs   {n_nodes:,} nodes   {n_edges:,} edges")
    print(f"status    {dict(status)}")
    if graphs:
        print(f"mean      {n_nodes/len(graphs):.1f} nodes, {n_edges/len(graphs):.1f} edges per drug")
    print(f"top fragment labels: {dict(frag_labels.most_common(10))}")
    failed = [r for r in reports if r["status"] not in ("ok",)]
    if failed:
        print(f"\n{len(failed)} rejected; first few:")
        for r in failed[:5]:
            print(f"   {r['drugbank_id']}: {r['status']} {r.get('problems') or ''}")
    print(f"\nwrote {out / 'drug_internal_graphs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
