#!/usr/bin/env python3
"""Phase 7: export the GoG as tensors for the multi-level GNN.

Writes framework-neutral `.npz` arrays plus a JSON manifest, and — when
torch_geometric is installed — a ready-to-load `gog_pyg.pt` holding the root
`HeteroData` and a list of per-owner internal `Data` objects.

Layout:

    root_nodes.npz         node ids, types, per-type index
    root_edges.npz         COO edge index per relation, for train / val / test
    drug_graphs.npz        one block per drug: x, edge_index, edge_attr, owner
    pathway_graphs.npz     one block per pathway, same shape
    cross_edges.npz        owner-level and node-level cross-context edges
    manifest.json          dims, vocabularies, per-type offsets, counts

The root graph's node features are left as learned embeddings — the manifest
gives the per-type counts to size them — except for drug and pathway owners,
whose vectors come up from their internal graph through whatever pooling the
model uses. `molecule_ref` nodes are flagged `exclude_from_pooling`, so pooling
must skip them: they are reference anchors for cross-context edges, not atoms.

    python scripts/09_export_tensors.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog import features as F
from gog.config import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", choices=["subset", "balanced"], default="subset",
                    help="which split files to export edge indices from")
    args = ap.parse_args()

    conf = cfg()
    out = conf["paths"]["out_dir"]
    export = out / "tensors"
    export.mkdir(parents=True, exist_ok=True)

    gog_path = out / "biomedical_gog.json"
    if not gog_path.exists():
        print("run scripts/07_integrate.py first", file=sys.stderr)
        return 1
    doc = json.loads(gog_path.read_text())

    # ------------------------------------------------------------ root nodes
    # The balanced splits span the hybrid universe (~9.6k nodes), not just the
    # GoG subset (~2.5k). Indexing against the subset would silently drop every
    # edge with an endpoint outside it — about 80k of them — so the node
    # universe has to match the split files being exported.
    root_nodes = doc["root_graph"]["nodes"]
    if args.splits == "balanced":
        hybrid_nodes_path = out / "hybrid_root_nodes.json"
        if not hybrid_nodes_path.exists():
            print("balanced splits need hybrid_root_nodes.json; "
                  "run scripts/10_make_hybrid_splits.py", file=sys.stderr)
            return 1
        root_nodes = json.loads(hybrid_nodes_path.read_text())["nodes"]
        print(f"node universe: {len(root_nodes):,} (hybrid)")
    else:
        print(f"node universe: {len(root_nodes):,} (GoG subset)")

    by_type: dict[str, list[str]] = defaultdict(list)
    for n in root_nodes:
        by_type[n["type"]].append(n["id"])
    local_index: dict[str, tuple[str, int]] = {}
    for t, ids in by_type.items():
        for i, nid in enumerate(sorted(ids)):
            local_index[nid] = (t, i)

    np.savez_compressed(
        export / "root_nodes.npz",
        ids=np.array([n["id"] for n in root_nodes], dtype=object),
        types=np.array([n["type"] for n in root_nodes], dtype=object),
        **{f"type__{t}": np.array(sorted(ids), dtype=object) for t, ids in by_type.items()},
    )

    # ------------------------------------------------------------ root edges
    prefix = "balanced_" if args.splits == "balanced" else ""
    splits = {}
    for name, fname in (("train", f"{prefix}train_facts.json"),
                        ("validation", f"{prefix}validation_facts.json"),
                        ("test", f"{prefix}test_facts.json")):
        p = out / fname
        if p.exists():
            splits[name] = json.loads(p.read_text())["facts"]
    if not splits:
        print(f"no {args.splits} split files found; run scripts/03_make_splits.py "
              f"or scripts/11_make_balanced_splits.py", file=sys.stderr)
        return 1

    edge_arrays: dict[str, np.ndarray] = {}
    rel_counts: dict[str, dict[str, int]] = {}
    skipped: dict[str, int] = {}
    for split, facts in splits.items():
        per_rel: dict[str, list[tuple[int, int]]] = defaultdict(list)
        missing = 0
        for f in facts:
            h, t = f["head"], f["tail"]
            if h not in local_index or t not in local_index:
                missing += 1
                continue
            ht, hi = local_index[h]
            tt, ti = local_index[t]
            per_rel[f"{ht}__{f['relation']}__{tt}"].append((hi, ti))
        rel_counts[split] = {k: len(v) for k, v in sorted(per_rel.items())}
        skipped[split] = missing
        for key, pairs in per_rel.items():
            arr = np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), np.int64)
            edge_arrays[f"{split}__{key}"] = arr
    np.savez_compressed(export / "root_edges.npz", **edge_arrays)

    # --------------------------------------------------------- drug graphs
    drug_graphs = doc["internal_graphs"]["drug"]
    frag_vocab = F.fragment_vocabulary(drug_graphs)
    frag_dim = len(frag_vocab) + len(F.FRAGMENT_NUMERIC)
    drug_dim = max(F.ATOM_FEATURE_DIM, frag_dim)

    drug_blocks: dict[str, np.ndarray] = {}
    drug_owners: list[str] = []
    for g in drug_graphs:
        # Atom and fragment vectors differ in length, so each is placed in a
        # zero-padded row of width drug_dim and tagged by a node-kind column.
        nodes = [n for n in g["nodes"] if n["type"] != "molecule_ref"]
        order = {n["id"]: i for i, n in enumerate(nodes)}
        x = np.zeros((len(nodes), drug_dim + 2), dtype=np.float32)
        for i, n in enumerate(nodes):
            attrs = n.get("attrs") or {}
            if n["type"] == "atom":
                vec = F.atom_features(attrs)
                x[i, : len(vec)] = vec
                x[i, drug_dim] = 1.0
            else:
                vec = F.fragment_features(attrs, frag_vocab)
                x[i, : len(vec)] = vec
                x[i, drug_dim + 1] = 1.0

        pairs, eattr = [], []
        for e in g["edges"]:
            if e["source"] not in order or e["target"] not in order:
                continue
            i, j = order[e["source"]], order[e["target"]]
            bf = F.bond_features(e.get("attrs") or {})
            pairs += [(i, j), (j, i)]            # molecular graphs are undirected
            eattr += [bf, bf]

        gid = g["graph_id"]
        drug_blocks[f"{gid}__x"] = x
        drug_blocks[f"{gid}__edge_index"] = (
            np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), np.int64))
        drug_blocks[f"{gid}__edge_attr"] = (
            np.array(eattr, dtype=np.float32) if eattr
            else np.zeros((0, len(F.BOND_TYPE_VOCAB) + 1), np.float32))
        drug_owners.append(f"{gid}|{g['owner_root_id']}")
    drug_blocks["__owners"] = np.array(drug_owners, dtype=object)
    np.savez_compressed(export / "drug_graphs.npz", **drug_blocks)

    # ------------------------------------------------------ pathway graphs
    pathway_graphs = doc["internal_graphs"]["pathway"]
    pw_dim = max(F.PATHWAY_ENTITY_FEATURE_DIM, F.REACTION_FEATURE_DIM)
    PW_EDGE_TYPES = ["input_to", "produces", "catalyzed_by", "activated_by",
                     "inhibited_by", "precedes", "set_member"]

    pw_blocks: dict[str, np.ndarray] = {}
    pw_owners: list[str] = []
    for g in pathway_graphs:
        nodes = g["nodes"]
        order = {n["id"]: i for i, n in enumerate(nodes)}
        x = np.zeros((len(nodes), pw_dim + 4), dtype=np.float32)
        for i, n in enumerate(nodes):
            attrs = n.get("attrs") or {}
            if n["type"] == "reaction_event":
                vec = F.reaction_features(attrs)
                x[i, pw_dim] = 1.0
            else:
                vec = F.pathway_entity_features(attrs)
                x[i, pw_dim + {"molecular_state": 1, "complex_state": 2,
                               "set_state": 3}.get(n["type"], 1)] = 1.0
            x[i, : len(vec)] = vec

        pairs, etypes = [], []
        for e in g["edges"]:
            if e["source"] not in order or e["target"] not in order:
                continue
            pairs.append((order[e["source"]], order[e["target"]]))
            etypes.append(PW_EDGE_TYPES.index(e["type"]) if e["type"] in PW_EDGE_TYPES else 0)

        gid = g["graph_id"]
        pw_blocks[f"{gid}__x"] = x
        pw_blocks[f"{gid}__edge_index"] = (
            np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), np.int64))
        pw_blocks[f"{gid}__edge_type"] = np.array(etypes, dtype=np.int64)
        pw_owners.append(f"{gid}|{g['owner_root_id']}")
    pw_blocks["__owners"] = np.array(pw_owners, dtype=object)
    np.savez_compressed(export / "pathway_graphs.npz", **pw_blocks)

    # -------------------------------------------------------- cross edges
    cross = doc["cross_context_edges"]
    cross_arrays = {
        "source": np.array([e["source"] for e in cross], dtype=object),
        "target": np.array([e["target"] for e in cross], dtype=object),
        "type": np.array([e["type"] for e in cross], dtype=object),
        "method": np.array([str((e.get("evidence") or {}).get("method")) for e in cross],
                           dtype=object),
    }
    np.savez_compressed(export / "cross_edges.npz", **cross_arrays)

    manifest = {
        "splits_used": args.splits,
        "root": {
            "node_type_counts": {t: len(ids) for t, ids in sorted(by_type.items())},
            "relation_keys": sorted({k.split("__", 1)[1] for k in edge_arrays}),
            "edges_per_split": {s: sum(v.values()) for s, v in rel_counts.items()},
            "edges_skipped_unknown_node": skipped,
            "relations_per_split": rel_counts,
        },
        "drug": {
            "n_graphs": len(drug_graphs),
            "feature_dim": drug_dim + 2,
            "atom_feature_dim": F.ATOM_FEATURE_DIM,
            "fragment_feature_dim": frag_dim,
            "fragment_vocabulary": frag_vocab,
            "kind_flag_columns": {"atom": drug_dim, "fragment": drug_dim + 1},
            "edge_attr_dim": len(F.BOND_TYPE_VOCAB) + 1,
            "note": "molecule_ref nodes are excluded from this export; "
                    "they exist only as cross-context anchors",
        },
        "pathway": {
            "n_graphs": len(pathway_graphs),
            "feature_dim": pw_dim + 4,
            "entity_feature_dim": F.PATHWAY_ENTITY_FEATURE_DIM,
            "reaction_feature_dim": F.REACTION_FEATURE_DIM,
            "edge_types": PW_EDGE_TYPES,
            "kind_flag_columns": {"reaction": pw_dim, "molecular_state": pw_dim + 1,
                                  "complex_state": pw_dim + 2, "set_state": pw_dim + 3},
        },
        "cross_context": {
            "n_edges": len(cross),
            "type_counts": dict(Counter(e["type"] for e in cross)),
            "method_counts": dict(Counter(
                str((e.get("evidence") or {}).get("method")) for e in cross)),
        },
        "feature_names": {
            "atom": F.ATOM_FEATURE_NAMES,
            "pathway_entity": F.PATHWAY_ENTITY_FEATURE_NAMES,
            "reaction": F.REACTION_FEATURE_NAMES,
        },
    }
    (export / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=" * 70)
    print("TENSOR EXPORT")
    print("=" * 70)
    print(f"root nodes       {len(root_nodes):,}  {manifest['root']['node_type_counts']}")
    print(f"root edges       {manifest['root']['edges_per_split']}")
    if any(skipped.values()):
        print(f"  SKIPPED (endpoint not in node universe): {skipped}")
    print(f"drug graphs      {len(drug_graphs):,}  feature dim {drug_dim + 2}"
          f"  ({len(frag_vocab)} fragment labels)")
    print(f"pathway graphs   {len(pathway_graphs):,}  feature dim {pw_dim + 4}")
    print(f"cross edges      {len(cross):,}  {manifest['cross_context']['type_counts']}")
    print(f"\nwrote {export}/")

    try:
        import torch
        from torch_geometric.data import Data, HeteroData
    except ImportError:
        print("\ntorch_geometric not installed; .npz + manifest are complete.")
        print("For the .pt bundle: pip install torch torch-geometric")
        return 0

    data = HeteroData()
    for t, ids in sorted(by_type.items()):
        data[t].num_nodes = len(ids)
    for key, arr in edge_arrays.items():
        split, rel = key.split("__", 1)
        src, relation, dst = rel.split("__")
        store_key = (src, f"{relation}__{split}", dst)
        data[store_key].edge_index = torch.from_numpy(arr)

    drug_data = []
    for g in drug_graphs:
        gid = g["graph_id"]
        drug_data.append(Data(
            x=torch.from_numpy(drug_blocks[f"{gid}__x"]),
            edge_index=torch.from_numpy(drug_blocks[f"{gid}__edge_index"]),
            edge_attr=torch.from_numpy(drug_blocks[f"{gid}__edge_attr"]),
            graph_id=gid, owner=g["owner_root_id"],
        ))
    pathway_data = []
    for g in pathway_graphs:
        gid = g["graph_id"]
        pathway_data.append(Data(
            x=torch.from_numpy(pw_blocks[f"{gid}__x"]),
            edge_index=torch.from_numpy(pw_blocks[f"{gid}__edge_index"]),
            edge_type=torch.from_numpy(pw_blocks[f"{gid}__edge_type"]),
            graph_id=gid, owner=g["owner_root_id"],
        ))

    torch.save({"root": data, "drug_graphs": drug_data,
                "pathway_graphs": pathway_data, "manifest": manifest},
               export / "gog_pyg.pt")
    print(f"wrote {export / 'gog_pyg.pt'} (PyG HeteroData + internal Data lists)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
