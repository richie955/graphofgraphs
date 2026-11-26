#!/usr/bin/env python3
"""Hybrid splits: a large training set, evaluated on GoG-relevant held-out facts.

The constraint in this project only ever applied to the *held-out* sets — they have
to contain facts the internal graphs can speak to, or the evaluation measures
nothing. Training data has no such constraint, so there is no reason to throw away
the 200k facts in the flat files.

So:

    validation / test   the subset splits, unchanged (100% GoG-relevant)
    train               the subset's train edges, plus every flat-file fact that
                        does not collide with a held-out edge in either direction

The flat files' own train/validation/test boundaries are ignored on purpose: their
held-out sets are not ours, so all 200k facts are candidate training data. What
matters is only that nothing in our validation or test set leaks in.

Row indices are mapped back to source identifiers through kg.csv, which is what the
flat files dropped. Nodes outside the GoG subset are kept and simply have no
internal graph — they get a learned embedding, which is how a real deployment
would work anyway.

    python scripts/10_make_hybrid_splits.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog import splits as sp
from gog.config import REPO_ROOT, cfg


def fact_id(head: str, relation: str, tail: str) -> str:
    return hashlib.sha256(f"{head}|{relation}|{tail}".encode()).hexdigest()[:16]


def undirected(head: str, relation: str, tail: str) -> tuple[str, str, str]:
    a, b = sorted((head, tail))
    return (a, relation, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", nargs="*", default=["train-rich.json", "validation_data.json",
                                                  "test_data.json"],
                    help="flat fact files to draw extra training data from")
    args = ap.parse_args()

    conf = cfg()
    out, interim = conf["paths"]["out_dir"], conf["paths"]["interim_dir"]

    for needed in ("train_facts.json", "validation_facts.json", "test_facts.json"):
        if not (out / needed).exists():
            print(f"missing {out / needed}; run scripts/03_make_splits.py first",
                  file=sys.stderr)
            return 1

    subset = {s: json.loads((out / f"{s}_facts.json").read_text())["facts"]
              for s in ("train", "validation", "test")}
    print(f"subset splits: train {len(subset['train']):,}, "
          f"validation {len(subset['validation']):,}, test {len(subset['test']):,}")

    # Held-out keys that nothing may duplicate, in either direction.
    held_out_keys = {
        undirected(f["head"], f["relation"], f["tail"])
        for s in ("validation", "test") for f in subset[s]
    }
    print(f"held-out undirected keys to protect: {len(held_out_keys):,}")

    # ---- index -> source id, rebuilt from kg.csv --------------------------
    kg = conf["paths"]["primekg_csv"]
    if not Path(kg).exists():
        print(f"need {kg} to map row indices back to source ids", file=sys.stderr)
        return 1
    print(f"\nbuilding row-index -> source-id map from {kg} ...")
    index_map = sp.index_to_source_id(kg)
    print(f"  {len(index_map):,} indices mapped")

    # ---- absorb the flat files -------------------------------------------
    train_keys = {undirected(f["head"], f["relation"], f["tail"]) for f in subset["train"]}
    hybrid: list[dict] = list(subset["train"])
    stats = Counter()
    new_nodes: dict[str, dict] = {}

    for fname in args.flat:
        path = REPO_ROOT / fname
        if not path.exists():
            print(f"  skipping missing {fname}")
            continue
        facts = json.loads(path.read_text())["facts"]
        print(f"\n{fname}: {len(facts):,} facts")
        added = 0
        for f in facts:
            h_meta = index_map.get(str(f["head"]))
            t_meta = index_map.get(str(f["tail"]))
            if not h_meta or not t_meta:
                stats["unmapped_index"] += 1
                continue
            head = f"root:{h_meta['type']}:{h_meta['id']}"
            tail = f"root:{t_meta['type']}:{t_meta['id']}"
            key = undirected(head, f["relation"], tail)
            if key in held_out_keys:
                stats["dropped_collides_with_held_out"] += 1
                continue
            if key in train_keys:
                stats["already_present"] += 1
                continue
            train_keys.add(key)
            for nid, meta in ((head, h_meta), (tail, t_meta)):
                if nid not in new_nodes:
                    new_nodes[nid] = {"id": nid, "type": meta["type"],
                                      "name": meta["name"],
                                      "attrs": {"source": meta["source"],
                                                "source_id": meta["id"]}}
            hybrid.append({
                "fact_id": fact_id(head, f["relation"], tail),
                "head": head, "head_id": h_meta["id"], "head_name": h_meta["name"],
                "head_type": h_meta["type"], "head_source": h_meta["source"],
                "relation": f["relation"],
                "display_relation": f.get("display_relation"),
                "tail": tail, "tail_id": t_meta["id"], "tail_name": t_meta["name"],
                "tail_type": t_meta["type"], "tail_source": t_meta["source"],
                "split": "train",
                "in_gog_root_universe": False,
                "origin": fname,
            })
            added += 1
        print(f"  added {added:,} new training edges")
        stats["added"] += added

    # ---- verify ------------------------------------------------------------
    hybrid_keys = {undirected(f["head"], f["relation"], f["tail"]) for f in hybrid}
    leak_val = sum(
        1 for f in subset["validation"]
        if undirected(f["head"], f["relation"], f["tail"]) in hybrid_keys)
    leak_test = sum(
        1 for f in subset["test"]
        if undirected(f["head"], f["relation"], f["tail"]) in hybrid_keys)

    root_doc = json.loads((out / "root_graph.json").read_text())
    subset_node_ids = {n["id"] for n in root_doc["nodes"]}
    all_nodes = {n["id"]: n for n in root_doc["nodes"]}
    for nid, n in new_nodes.items():
        all_nodes.setdefault(nid, n)
    outside = len(all_nodes) - len(subset_node_ids)

    owner_ids = set()
    seeds_path = interim / "seeds.json"
    if seeds_path.exists():
        seeds = json.loads(seeds_path.read_text())
        owner_ids = {f"root:drug:{d}" for d in seeds.get("drugs", [])} | \
                    {f"root:pathway:{p}" for p in seeds.get("pathways", [])}

    def touches_owner(f: dict) -> bool:
        return f["head"] in owner_ids or f["tail"] in owner_ids

    payload = {
        "metadata": {
            "split": "train",
            "total_facts": len(hybrid),
            "unique_undirected_edges": len(hybrid_keys),
            "from_subset": len(subset["train"]),
            "from_flat_files": stats["added"],
            "sources": args.flat,
            "dropped_collides_with_held_out": stats["dropped_collides_with_held_out"],
            "dropped_unmapped_index": stats["unmapped_index"],
            "already_present": stats["already_present"],
            "facts_touching_an_internal_graph": sum(1 for f in hybrid if touches_owner(f)),
            "leakage_check": {"validation_in_train": leak_val, "test_in_train": leak_test},
            "note": "validation and test are unchanged from 03_make_splits.py; "
                    "only the training set grows. Nodes outside the GoG subset "
                    "have no internal graph and take a learned embedding.",
        },
        "facts": hybrid,
    }
    (out / "hybrid_train_facts.json").write_text(json.dumps(payload, indent=2))

    node_payload = {
        "metadata": {
            "n_nodes": len(all_nodes),
            "in_gog_subset": len(subset_node_ids),
            "outside_gog_subset": outside,
            "node_type_counts": dict(Counter(n["type"] for n in all_nodes.values())),
            "note": "embedding table size for the hybrid setup",
        },
        "nodes": list(all_nodes.values()),
    }
    (out / "hybrid_root_nodes.json").write_text(json.dumps(node_payload, indent=2))

    m = payload["metadata"]
    print("\n" + "=" * 72)
    print("HYBRID SPLITS")
    print("=" * 72)
    print(f"train           {len(hybrid):>10,} facts "
          f"({m['from_subset']:,} subset + {m['from_flat_files']:,} flat)")
    print(f"validation      {len(subset['validation']):>10,} facts   (unchanged, 100% GoG)")
    print(f"test            {len(subset['test']):>10,} facts   (unchanged, 100% GoG)")
    print(f"train touching an internal graph: {m['facts_touching_an_internal_graph']:,}")
    print(f"\nnodes           {len(all_nodes):>10,} "
          f"({len(subset_node_ids):,} in subset + {outside:,} outside)")
    print(f"node types      {node_payload['metadata']['node_type_counts']}")
    print(f"\ndropped (would leak held-out): {m['dropped_collides_with_held_out']:,}")
    print(f"unmapped row indices:          {m['dropped_unmapped_index']:,}")
    print("=" * 72)
    if leak_val or leak_test:
        print(f"LEAKAGE: validation {leak_val}, test {leak_test} — do not use", file=sys.stderr)
        return 2
    print("leakage check PASSED: no validation or test edge appears in train")
    print(f"\nwrote {out / 'hybrid_train_facts.json'}")
    print(f"wrote {out / 'hybrid_root_nodes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
