#!/usr/bin/env python3
"""Phase 3: build a reaction-level internal graph for every pathway in the subset.

Takes the pathway nodes from root_graph.json — not just the coverage-check seeds —
because 1-hop expansion pulls in pathways that were never sized, and a pathway
node with no internal graph is a wasted owner.

Network-bound and cached: a re-run costs nothing, and interrupting it loses only
the pathway in flight.

    python scripts/05_build_pathway_graphs.py
    python scripts/05_build_pathway_graphs.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from gog.config import cfg
from gog.pathway_graphs import build_pathway_graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="first N pathways only (0 = all)")
    args = ap.parse_args()

    conf = cfg()
    interim, out = conf["paths"]["interim_dir"], conf["paths"]["out_dir"]
    max_nodes = conf["subset"]["max_pathway_nodes"]
    min_nodes = conf["subset"].get("min_pathway_nodes", 1)

    root_path = out / "root_graph.json"
    if not root_path.exists():
        print("run scripts/02_build_root_graph.py first", file=sys.stderr)
        return 1

    doc = json.loads(root_path.read_text())
    pathways = sorted({
        n["attrs"]["source_id"] for n in doc["nodes"]
        if n["type"] == "pathway" and str(n["attrs"].get("source_id", "")).startswith("R-")
    })
    if args.limit:
        pathways = pathways[: args.limit]
    print(f"building {len(pathways)} pathway internal graphs "
          f"(node range [{min_nodes}, {max_nodes}])")

    graphs = []
    reports = []
    status: Counter[str] = Counter()
    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()

    for st_id in tqdm(pathways, unit="pathway"):
        try:
            b = build_pathway_graph(st_id, max_nodes=max_nodes, min_nodes=min_nodes)
        except Exception as exc:                      # one bad pathway must not kill the run
            reports.append({"st_id": st_id, "status": "error", "error": repr(exc)})
            status["error"] += 1
            continue
        reports.append(b.report)
        status[b.report.get("status", "unknown")] += 1
        if b.graph is not None:
            graphs.append(b.graph)
            for n in b.graph.nodes:
                node_types[n.type] += 1
            for e in b.graph.edges:
                edge_types[e.type] += 1

    payload = {
        "metadata": {
            "n_graphs": len(graphs),
            "status_counts": dict(status),
            "max_pathway_nodes": max_nodes,
            "min_pathway_nodes": min_nodes,
            "node_type_counts": dict(node_types),
            "edge_type_counts": dict(edge_types),
            "builder": "reactome_content_service/v1",
        },
        "graphs": [asdict(g) for g in graphs],
    }
    (out / "pathway_internal_graphs.json").write_text(json.dumps(payload, indent=2))
    (interim / "pathway_graph_reports.json").write_text(json.dumps(reports, indent=2))

    n_nodes = sum(len(g.nodes) for g in graphs)
    n_edges = sum(len(g.edges) for g in graphs)
    print("\n" + "=" * 66)
    print(f"built   {len(graphs):,} graphs   {n_nodes:,} nodes   {n_edges:,} edges")
    print(f"status  {dict(status)}")
    if graphs:
        print(f"mean    {n_nodes/len(graphs):.1f} nodes, {n_edges/len(graphs):.1f} edges per pathway")
    print(f"nodes   {dict(node_types)}")
    print(f"edges   {dict(edge_types)}")
    print(f"\nwrote {out / 'pathway_internal_graphs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
