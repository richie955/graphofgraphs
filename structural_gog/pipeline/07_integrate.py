#!/usr/bin/env python3
"""Phase 5: combine every component into biomedical_gog.json.

Assembles the root graph, both families of internal graphs, the owner links and
the cross-context edges into the single envelope defined in `schema.envelope()`,
then runs `schema.validate_envelope()` before writing. A document that fails
validation is still written, to `biomedical_gog.invalid.json`, so the problems
can be inspected rather than guessed at.

    python scripts/07_integrate.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog import schema
from gog.config import cfg
from gog.schema import CrossContextEdge, Edge, InternalGraph, Node


def as_nodes(rows: list[dict]) -> list[Node]:
    return [Node(id=r["id"], type=r["type"], name=r.get("name"),
                 attrs=r.get("attrs") or {}) for r in rows]


def as_edges(rows: list[dict]) -> list[Edge]:
    return [Edge(source=r["source"], target=r["target"], type=r["type"],
                 attrs=r.get("attrs") or {}) for r in rows]


def as_internal(rows: list[dict]) -> list[InternalGraph]:
    return [
        InternalGraph(
            graph_id=g["graph_id"], owner_root_id=g["owner_root_id"], kind=g["kind"],
            nodes=as_nodes(g["nodes"]), edges=as_edges(g["edges"]),
            provenance=g.get("provenance") or {},
        )
        for g in rows
    ]


def main() -> int:
    conf = cfg()
    out = conf["paths"]["out_dir"]

    required = ["root_graph.json", "drug_internal_graphs.json",
                "pathway_internal_graphs.json", "cross_context_edges.json"]
    missing = [r for r in required if not (out / r).exists()]
    if missing:
        print(f"missing: {', '.join(missing)}", file=sys.stderr)
        return 1

    root = json.loads((out / "root_graph.json").read_text())
    drug_doc = json.loads((out / "drug_internal_graphs.json").read_text())
    pw_doc = json.loads((out / "pathway_internal_graphs.json").read_text())
    cross_doc = json.loads((out / "cross_context_edges.json").read_text())

    root_nodes = as_nodes(root["nodes"])
    root_edges = as_edges(root["edges"])
    drug_graphs = as_internal(drug_doc["graphs"])
    pathway_graphs = as_internal(pw_doc["graphs"])

    # Owner links: root node -> internal graph. Only for graphs whose owner is
    # actually in the root graph, or the envelope would not validate.
    root_ids = {n.id for n in root_nodes}
    owner_links: list[Edge] = []
    orphaned: list[str] = []
    for g in drug_graphs + pathway_graphs:
        if g.owner_root_id in root_ids:
            owner_links.append(Edge(source=g.owner_root_id, target=g.graph_id,
                                    type="owns_internal_graph",
                                    attrs={"kind": g.kind,
                                           "n_nodes": len(g.nodes),
                                           "n_edges": len(g.edges)}))
        else:
            orphaned.append(g.graph_id)

    cross_edges = [
        CrossContextEdge(source=e["source"], target=e["target"], type=e["type"],
                         evidence=e.get("evidence") or {})
        for e in cross_doc["cross_context_edges"]
    ]

    # Drop internal graphs that no owner link reached, then drop cross edges
    # whose endpoints no longer exist. Keeping either would produce a document
    # that references nodes it does not contain.
    if orphaned:
        keep = set(orphaned)
        drug_graphs = [g for g in drug_graphs if g.graph_id not in keep]
        pathway_graphs = [g for g in pathway_graphs if g.graph_id not in keep]

    present = set(root_ids)
    for g in drug_graphs + pathway_graphs:
        present |= {n.id for n in g.nodes}
    dropped_cross = [e for e in cross_edges
                     if e.source not in present or e.target not in present]
    cross_edges = [e for e in cross_edges
                   if e.source in present and e.target in present]

    internal_nodes = sum(len(g.nodes) for g in drug_graphs + pathway_graphs)
    internal_edges = sum(len(g.edges) for g in drug_graphs + pathway_graphs)

    metadata = {
        "built": date.today().isoformat(),
        "schema_version": schema.SCHEMA_VERSION,
        "root": {
            "n_nodes": len(root_nodes),
            "n_edges": len(root_edges),
            "node_type_counts": dict(Counter(n.type for n in root_nodes)),
            "relation_counts": dict(Counter(e.type for e in root_edges).most_common()),
        },
        "internal": {
            "n_drug_graphs": len(drug_graphs),
            "n_pathway_graphs": len(pathway_graphs),
            "n_nodes": internal_nodes,
            "n_edges": internal_edges,
            "drug_node_types": dict(Counter(
                n.type for g in drug_graphs for n in g.nodes)),
            "drug_edge_types": dict(Counter(
                e.type for g in drug_graphs for e in g.edges)),
            "pathway_node_types": dict(Counter(
                n.type for g in pathway_graphs for n in g.nodes)),
            "pathway_edge_types": dict(Counter(
                e.type for g in pathway_graphs for e in g.edges)),
        },
        "owner_links": len(owner_links),
        "cross_context": {
            "n_edges": len(cross_edges),
            "type_counts": dict(Counter(e.type for e in cross_edges)),
            "method_counts": dict(Counter(
                str(e.evidence.get("method")) for e in cross_edges)),
            "dropped_unresolved_endpoints": len(dropped_cross),
            "leakage_filter": cross_doc["metadata"].get("leakage_filter"),
        },
        "orphaned_internal_graphs": orphaned,
        "structural_gain": {
            "note": "nodes and edges the internal graphs add beyond PrimeKG",
            "nodes_added": internal_nodes,
            "edges_added": internal_edges,
            "ratio_internal_to_root_nodes": round(internal_nodes / max(1, len(root_nodes)), 2),
        },
    }

    doc = schema.envelope(root_nodes, root_edges, drug_graphs, pathway_graphs,
                          owner_links, cross_edges, metadata)
    problems = schema.validate_envelope(doc)

    target = out / ("biomedical_gog.json" if not problems else "biomedical_gog.invalid.json")
    target.write_text(json.dumps(doc, indent=2))

    print("=" * 70)
    print("INTEGRATED GRAPH-OF-GRAPHS")
    print("=" * 70)
    print(f"root            {len(root_nodes):>8,} nodes  {len(root_edges):>8,} edges")
    print(f"drug graphs     {len(drug_graphs):>8,}")
    print(f"pathway graphs  {len(pathway_graphs):>8,}")
    print(f"internal total  {internal_nodes:>8,} nodes  {internal_edges:>8,} edges")
    print(f"owner links     {len(owner_links):>8,}")
    print(f"cross-context   {len(cross_edges):>8,}  {metadata['cross_context']['type_counts']}")
    if dropped_cross:
        print(f"  dropped {len(dropped_cross)} cross edges with unresolved endpoints")
    if orphaned:
        print(f"  dropped {len(orphaned)} internal graphs with no owner in the root graph")
    print("=" * 70)
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problems):")
        for p in problems[:15]:
            print(f"   - {p}")
        print(f"\nwrote {target} (invalid)")
        return 2
    print("validation passed")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
