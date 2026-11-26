#!/usr/bin/env python3
"""Export the GoG into the four-table schema the graphofgraphs model consumes.

Target contract: `docs/00_data_contract_and_preprocessing.md` in the model repo —
`entities`, `graphs`, `graph_nodes`, `relations`, `metadata`. Flat records only.
`biomedical_gog.json` stays as the construction artifact; none of its provenance,
atom indices or evidence dicts cross into the model file.

Decisions this implements (see SCHEMA_FOR_MODEL.md):

* **D1** pathway entities are keyed by Reactome stId, so one protein is one entity
  with occurrences in many pathway graphs. That is what Stage 7's identity
  synchronisation operates on.
* **D2** entities are NOT merged into their root counterparts. 94 of 124 pathway
  graphs hold two or more entities sharing one UniProt accession (TNF appears as a
  trimer, a receptor complex, a soluble form), so merging would either break the
  one-GraphNode-per-(entity, graph) invariant or collapse distinct molecular states.
  Identity ships relationally instead, as `same_entity_as` / `same_chemical_as`.
* **D3** `molecule_ref` anchors are dropped; `same_chemical_as` is retargeted to the
  root drug entity.
* **D4** entity pairs that span two splits are moved wholly into train, because the
  model's split unit is the unordered entity pair.

Only PrimeKG root facts carry held-out labels. Structural and cross-context edges are
all `train`: encoder input, never prediction targets (handoff §4).

    python scripts/12_export_to_model.py --splits balanced
    python scripts/12_export_to_model.py --splits subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog.config import cfg

ROOT_GRAPH = "ROOT"


def atom_type(attrs: dict) -> str:
    """Chemistry-bearing entity type for an atom.

    Their Stage 1 computes `b_entity = entity_embedding + type_embedding(type)`, so a
    type is a *shared trained parameter* while an entity embedding is private. Atoms
    have one occurrence and ~1.6 relations each, so their private embeddings never
    train — but a type shared by every `C, degree 1` atom in the corpus receives
    gradient from all 487 of them.

    The chemistry cannot go in `entity_id` instead: two methyl groups in one molecule
    would become the same entity in the same graph, which the one-GraphNode-per-
    (entity, graph) invariant forbids. Encoding it in `type` keeps ids unique and
    still shares the parameters.
    """
    el = attrs.get("element") or "X"
    parts = [f"atom_{el}", f"deg{int(attrs.get('degree') or 0)}"]
    if attrs.get("aromatic"):
        parts.append("arom")
    q = int(attrs.get("formal_charge") or 0)
    if q:
        parts.append(f"q{'p' if q > 0 else 'm'}{abs(q)}")
    if attrs.get("in_ring") and not attrs.get("aromatic"):
        parts.append("ring")
    return "_".join(parts)


def fragment_type(attrs: dict) -> str:
    """One type per functional-group / ring-system label, shared across molecules."""
    label = str(attrs.get("group_name") or "unknown")
    return f"frag_{label}"


def graph_id_for(owner_root_id: str) -> str:
    """root:drug:DB00945 -> G_drug_DB00945 ; root:pathway:R-HSA-1 -> G_pathway_R-HSA-1"""
    _, kind, source_id = owner_root_id.split(":", 2)
    return f"G_{kind}_{source_id}"


def rel_id(prefix: str, source: str, rtype: str, target: str) -> str:
    digest = hashlib.sha256(f"{source}|{rtype}|{target}".encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", choices=["balanced", "subset"], default="balanced")
    ap.add_argument("--out", default=None, help="output filename (default derives from --splits)")
    ap.add_argument("--coarse-types", action="store_true",
                    help="use plain 'atom'/'molecular_fragment' types instead of "
                         "chemistry-bearing ones (the original, weaker encoding)")
    ap.add_argument("--no-internal", action="store_true",
                    help="ABLATION: emit the root graph and its facts only — no internal "
                         "graphs, no cross-context edges. Identical entities, splits and "
                         "candidate universe, so running the same encoder on this file "
                         "and on the full one isolates what the structural level "
                         "contributes. This is the comparison that answers 'does the "
                         "graph-of-graphs help', which a ComplEx baseline cannot.")
    args = ap.parse_args()

    conf = cfg()
    out_dir = conf["paths"]["out_dir"]

    gog_path = out_dir / "biomedical_gog.json"
    if not gog_path.exists():
        print("run scripts/07_integrate.py first", file=sys.stderr)
        return 1
    doc = json.loads(gog_path.read_text())

    prefix = "balanced_" if args.splits == "balanced" else ""
    split_files = {s: out_dir / f"{prefix}{s}_facts.json"
                   for s in ("train", "validation", "test")}
    missing = [str(p) for p in split_files.values() if not p.exists()]
    if missing:
        print(f"missing split files: {', '.join(missing)}", file=sys.stderr)
        return 1
    splits = {s: json.loads(p.read_text())["facts"] for s, p in split_files.items()}

    # The balanced splits span the hybrid universe; the subset splits do not.
    if args.splits == "balanced":
        nodes_path = out_dir / "hybrid_root_nodes.json"
        if not nodes_path.exists():
            print("balanced export needs hybrid_root_nodes.json", file=sys.stderr)
            return 1
        root_nodes = json.loads(nodes_path.read_text())["nodes"]
    else:
        root_nodes = doc["root_graph"]["nodes"]
    print(f"root entities: {len(root_nodes):,} ({args.splits})")

    # ------------------------------------------------------------- entities
    entities: dict[str, dict] = {}

    def add_entity(eid: str, name: str | None, etype: str,
                   source: str | None = None, source_id: str | None = None) -> None:
        if eid in entities:
            return
        entities[eid] = {
            "entity_id": eid,
            "name": (name or eid)[:300],
            "type": etype,
            "source": source or "",
            "source_id": source_id or "",
        }

    for n in root_nodes:
        attrs = n.get("attrs") or {}
        add_entity(n["id"], n.get("name"), n["type"],
                   attrs.get("source"), attrs.get("source_id"))
    root_entity_ids = set(entities)

    # Drug internal nodes are unique to their molecule, so the node id is the
    # entity id. molecule_ref is dropped (D3).
    drug_graphs = [] if args.no_internal else doc["internal_graphs"]["drug"]
    drug_node_entity: dict[str, str] = {}
    for g in drug_graphs:
        for n in g["nodes"]:
            if n["type"] == "molecule_ref":
                continue
            drug_node_entity[n["id"]] = n["id"]
            attrs = n.get("attrs") or {}
            label = attrs.get("group_name") or n.get("name")
            if args.coarse_types:
                etype = n["type"]
            elif n["type"] == "atom":
                etype = atom_type(attrs)
            else:
                etype = fragment_type(attrs)
            add_entity(n["id"], label, etype, "RDKit", None)

    # Pathway nodes collapse onto their Reactome stId (D1).
    pathway_graphs = [] if args.no_internal else doc["internal_graphs"]["pathway"]
    pw_node_entity: dict[str, str] = {}
    for g in pathway_graphs:
        for n in g["nodes"]:
            st = (n.get("attrs") or {}).get("reactome_st_id")
            if not st:
                continue
            eid = f"reactome:{st}"
            pw_node_entity[n["id"]] = eid
            add_entity(eid, n.get("name"), n["type"], "REACTOME", st)

    print(f"entities: {len(entities):,} total "
          f"({len(root_entity_ids):,} root, {len(drug_node_entity):,} drug internal, "
          f"{len({v for v in pw_node_entity.values()}):,} pathway)")

    # --------------------------------------------------------------- graphs
    graphs = [{"graph_id": ROOT_GRAPH, "type": "root"}]
    owner_of_graph: dict[str, str] = {}
    graph_of_internal: dict[str, str] = {}      # internal graph_id (mine) -> model graph id
    skipped_owner = 0
    for g in drug_graphs + pathway_graphs:
        owner = g["owner_root_id"]
        if owner not in root_entity_ids:
            skipped_owner += 1
            continue
        gid = graph_id_for(owner)
        graphs.append({"graph_id": gid, "type": "internal", "parent_entity_id": owner})
        owner_of_graph[gid] = owner
        graph_of_internal[g["graph_id"]] = gid
    if skipped_owner:
        print(f"  skipped {skipped_owner} internal graphs whose owner is not a root entity")
    print(f"graphs: 1 ROOT + {len(graphs)-1} internal")

    # ---------------------------------------------------------- graph_nodes
    graph_nodes: dict[tuple[str, str], dict] = {}

    def add_node(eid: str, gid: str) -> str:
        key = (eid, gid)
        if key not in graph_nodes:
            graph_nodes[key] = {"node_id": f"{eid}@{gid}", "entity_id": eid, "graph_id": gid}
        return graph_nodes[key]["node_id"]

    for eid in root_entity_ids:
        add_node(eid, ROOT_GRAPH)
    for g in drug_graphs:
        gid = graph_of_internal.get(g["graph_id"])
        if not gid:
            continue
        for n in g["nodes"]:
            if n["id"] in drug_node_entity:
                add_node(drug_node_entity[n["id"]], gid)
    for g in pathway_graphs:
        gid = graph_of_internal.get(g["graph_id"])
        if not gid:
            continue
        for n in g["nodes"]:
            if n["id"] in pw_node_entity:
                add_node(pw_node_entity[n["id"]], gid)
    print(f"graph_nodes: {len(graph_nodes):,}")

    # ------------------------------------------------------------ relations
    relations: list[dict] = []
    seen_rel: set[tuple[str, str, str]] = set()
    dropped = Counter()

    def emit(source_node: str, rtype: str, target_node: str, *,
             split: str, display: str | None, prefix: str) -> None:
        key = (source_node, rtype, target_node)
        if key in seen_rel:
            dropped["duplicate_relation"] += 1
            return
        seen_rel.add(key)
        relations.append({
            "relation_id": rel_id(prefix, source_node, rtype, target_node),
            "source_node_id": source_node,
            "type": rtype,
            "target_node_id": target_node,
            "display_relation": display or "",
            "split": split,
        })

    # D4: the model's split unit is the unordered entity pair, so a pair that spans
    # two splits is a contract error. Move every such pair wholly into train.
    pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s, facts in splits.items():
        for f in facts:
            a, b = sorted((f["head"], f["tail"]))
            pair_splits[(a, b)].add(s)
    conflicted = {p for p, ss in pair_splits.items() if len(ss) > 1}
    print(f"entity pairs spanning >1 split: {len(conflicted)} -> forced to train")

    for s, facts in splits.items():
        for f in facts:
            h, t = f["head"], f["tail"]
            if h not in root_entity_ids or t not in root_entity_ids:
                dropped["root_fact_unknown_entity"] += 1
                continue
            pair = tuple(sorted((h, t)))
            split = "train" if pair in conflicted else s
            emit(f"{h}@{ROOT_GRAPH}", f["relation"], f"{t}@{ROOT_GRAPH}",
                 split=split, display=f.get("display_relation"), prefix="R")

    # Structural edges: encoder input only, never targets.
    for g in drug_graphs:
        gid = graph_of_internal.get(g["graph_id"])
        if not gid:
            continue
        for e in g["edges"]:
            se, te = drug_node_entity.get(e["source"]), drug_node_entity.get(e["target"])
            if not se or not te:
                dropped["drug_edge_dropped_endpoint"] += 1
                continue
            emit(f"{se}@{gid}", e["type"], f"{te}@{gid}", split="train",
                 display=(e.get("attrs") or {}).get("bond_type"), prefix="D")

    for g in pathway_graphs:
        gid = graph_of_internal.get(g["graph_id"])
        if not gid:
            continue
        for e in g["edges"]:
            se, te = pw_node_entity.get(e["source"]), pw_node_entity.get(e["target"])
            if not se or not te:
                dropped["pathway_edge_dropped_endpoint"] += 1
                continue
            if se == te:
                # Two states collapsed onto one stId would make a self-loop; the
                # stId keying makes this impossible in practice, but never emit one.
                dropped["pathway_self_loop"] += 1
                continue
            emit(f"{se}@{gid}", e["type"], f"{te}@{gid}", split="train",
                 display=(e.get("attrs") or {}).get("role"), prefix="P")

    # Cross-context edges. Endpoints must resolve to an existing GraphNode.
    def resolve_cross(nid: str) -> str | None:
        if nid in root_entity_ids:
            return f"{nid}@{ROOT_GRAPH}"
        if nid in drug_node_entity:
            owner = nid.split("::", 1)[0]
            gid = graph_of_internal.get(owner)
            return f"{drug_node_entity[nid]}@{gid}" if gid else None
        if nid in pw_node_entity:
            owner = nid.split("::", 1)[0]
            gid = graph_of_internal.get(owner)
            return f"{pw_node_entity[nid]}@{gid}" if gid else None
        if nid.endswith("::mol"):
            # D3: molecule_ref dropped; same_chemical_as retargets to the root drug.
            db = nid.split(":")[1]
            root = f"root:drug:{db}"
            return f"{root}@{ROOT_GRAPH}" if root in root_entity_ids else None
        return None

    cross_edges = [] if args.no_internal else doc["cross_context_edges"]
    for e in cross_edges:
        s_node, t_node = resolve_cross(e["source"]), resolve_cross(e["target"])
        if not s_node or not t_node:
            dropped["cross_edge_unresolved"] += 1
            continue
        if s_node == t_node:
            dropped["cross_self_loop"] += 1
            continue
        emit(s_node, e["type"], t_node, split="train",
             display=(e.get("evidence") or {}).get("method"), prefix="X")

    by_split = Counter(r["split"] for r in relations)
    by_type = Counter(r["type"] for r in relations)
    print(f"relations: {len(relations):,}  {dict(by_split)}")
    if dropped:
        print(f"  dropped: {dict(dropped)}")

    # ------------------------------------------------------------- assemble
    payload = {
        "entities": sorted(entities.values(), key=lambda r: r["entity_id"]),
        "graphs": graphs,
        "graph_nodes": sorted(graph_nodes.values(), key=lambda r: r["node_id"]),
        "relations": relations,
        "metadata": {
            "dataset": "PrimeKG + PubChem/RDKit + Reactome structural GoG",
            "builder": "structural_gog/pipeline/12_export_to_model.py",
            "split_configuration": args.splits,
            "ablation": ("root_only (no internal graphs, no cross-context edges)"
                         if args.no_internal else "full graph-of-graphs"),
            "splits_are_source_provided": True,
            "counts": {
                "entities": len(entities),
                "graphs": len(graphs),
                "internal_graphs": len(graphs) - 1,
                "graph_nodes": len(graph_nodes),
                "relations": len(relations),
                "relations_by_split": dict(by_split),
                "relations_by_type": dict(by_type.most_common()),
            },
            "entity_type_counts": dict(Counter(e["type"] for e in entities.values())),
            "internal_graph_owner_types": ["drug", "pathway"],
            "internal_graph_statistics": [
                {"owner_type": "drug", "graphs_built": len(drug_graphs)},
                {"owner_type": "pathway", "graphs_built": len(pathway_graphs)},
            ],
            "structural_sources": {
                "drug": "PubChem SMILES -> RDKit ring systems + SMARTS functional groups",
                "pathway": "Reactome Content Service reactions and participants",
            },
            "held_out_scope": "PrimeKG root facts only; structural and cross-context "
                              "edges are train-split encoder channels, never targets",
            "pathway_entity_keying": "Reactome stId (shared across pathway graphs)",
            "drug_entity_typing": ("coarse (atom / molecular_fragment)" if args.coarse_types
                                   else "chemistry-bearing (element, degree, charge, "
                                        "aromaticity; one type per fragment label)"),
            "drug_typing_rationale": ("Stage 1 adds a shared type embedding to each "
                                      "private entity embedding. Atoms have one "
                                      "occurrence and ~1.6 relations each, so their "
                                      "private embeddings cannot train; the type "
                                      "carries the chemistry and is shared corpus-wide."),
            "identity_policy": "not merged into root entities; expressed via "
                               "same_entity_as / same_chemical_as relations",
            "entity_pairs_forced_to_train": len(conflicted),
            "dropped": dict(dropped),
            "known_limitation": "subset selection preceded fact splitting, so graph "
                                "construction saw edges later held out (handoff 3.4)",
        },
    }

    suffix = "_rootonly" if args.no_internal else ""
    name = args.out or f"gog_model_{args.splits}{suffix}.json"
    target = out_dir / name
    target.write_text(json.dumps(payload, separators=(",", ":")))
    size_mb = target.stat().st_size / 1e6

    print("\n" + "=" * 70)
    print("MODEL-INPUT FILE")
    print("=" * 70)
    print(f"entities      {len(entities):>9,}  {payload['metadata']['entity_type_counts']}")
    print(f"graphs        {len(graphs):>9,}")
    print(f"graph_nodes   {len(graph_nodes):>9,}")
    print(f"relations     {len(relations):>9,}  {dict(by_split)}")
    print(f"\nwrote {target}  ({size_mb:.1f} MB)")
    print("\nNEXT: validate with the model repo's own loader, then")
    print(f"  python -m training.train {name} --model compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
