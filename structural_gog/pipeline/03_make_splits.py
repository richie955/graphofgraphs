#!/usr/bin/env python3
"""Phase 1b: split the root graph's edges into train/validation/test.

Splits the GoG subset itself, not PrimeKG at large, so every held-out fact is one
the internal graphs can actually speak to. Differences from the earlier flat files:

* **Source identifiers are kept.** `head_id` is DB00945 / R-HSA-109582, so a fact
  joins to an internal graph without going back to kg.csv.
* **One record per edge.** PrimeKG stores both directions; keeping both would make
  every reported metric count each edge twice. The original direction is preserved
  on the surviving record, and the model adds inverse edges itself.
* **Stratified by relation**, so rare relations such as `indication` get a usable
  test count instead of 20.
* **No cold-start endpoints.** A val/test edge whose node appears nowhere in train
  is moved to train; an unseen node has no embedding to rank with.

    python scripts/03_make_splits.py
    python scripts/03_make_splits.py --test-frac 0.1 --val-frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog.config import cfg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-held-out", type=int, default=4,
                    help="relations with fewer unique edges than this stay wholly in train")
    return ap.parse_args()


def fact_id(head: str, relation: str, tail: str) -> str:
    return hashlib.sha256(f"{head}|{relation}|{tail}".encode()).hexdigest()[:16]


def main() -> int:
    args = parse_args()
    conf = cfg()
    out_dir, interim = conf["paths"]["out_dir"], conf["paths"]["interim_dir"]

    root_path = out_dir / "root_graph.json"
    if not root_path.exists():
        print(f"missing {root_path}; run scripts/02_build_root_graph.py first", file=sys.stderr)
        return 1

    doc = json.loads(root_path.read_text())
    nodes = {n["id"]: n for n in doc["nodes"]}
    seeds = json.loads((interim / "seeds.json").read_text()) if (interim / "seeds.json").exists() else {}
    owner_ids = {f"root:drug:{d}" for d in seeds.get("drugs", [])} | \
                {f"root:pathway:{p}" for p in seeds.get("pathways", [])}

    # ---------------------------------------------------- collapse directions
    seen: dict[tuple[str, str, str], dict] = {}
    dropped_mirror = 0
    for e in doc["edges"]:
        a, b = sorted((e["source"], e["target"]))
        key = (a, e["type"], b)
        if key in seen:
            dropped_mirror += 1
            continue
        seen[key] = e          # keep PrimeKG's original direction
    edges = list(seen.values())
    print(f"root graph: {len(doc['edges']):,} directed edges -> {len(edges):,} unique "
          f"({dropped_mirror:,} mirror copies collapsed)")

    # ------------------------------------------------------ stratified split
    rng = random.Random(args.seed)
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        by_rel[e["type"]].append(e)

    assign: dict[int, str] = {}
    held_back: list[str] = []
    for rel, group in sorted(by_rel.items()):
        group = sorted(group, key=lambda e: (e["source"], e["target"]))
        rng.shuffle(group)
        if len(group) < args.min_held_out:
            held_back.append(f"{rel} ({len(group)})")
            for e in group:
                assign[id(e)] = "train"
            continue
        n_val = max(1, int(round(len(group) * args.val_frac)))
        n_test = max(1, int(round(len(group) * args.test_frac)))
        for i, e in enumerate(group):
            assign[id(e)] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"

    # --------------------------------------------- no cold-start endpoints
    train_nodes = {e["source"] for e in edges if assign[id(e)] == "train"} | \
                  {e["target"] for e in edges if assign[id(e)] == "train"}
    moved = 0
    for e in edges:
        if assign[id(e)] == "train":
            continue
        if e["source"] not in train_nodes or e["target"] not in train_nodes:
            assign[id(e)] = "train"
            train_nodes.add(e["source"])
            train_nodes.add(e["target"])
            moved += 1
    if moved:
        print(f"moved {moved:,} held-out edges to train (endpoint unseen in train)")

    # ----------------------------------------------------------- serialise
    def record(e: dict, split: str) -> dict:
        h, t = nodes[e["source"]], nodes[e["target"]]
        return {
            "fact_id": fact_id(e["source"], e["type"], e["target"]),
            "head": e["source"],
            "head_id": h["attrs"].get("source_id"),
            "head_name": h.get("name"),
            "head_type": h["type"],
            "head_source": h["attrs"].get("source"),
            "relation": e["type"],
            "display_relation": e["attrs"].get("display_relation"),
            "tail": e["target"],
            "tail_id": t["attrs"].get("source_id"),
            "tail_name": t.get("name"),
            "tail_type": t["type"],
            "tail_source": t["attrs"].get("source"),
            "split": split,
            "in_gog_root_universe": True,
            "has_internal_graph": {
                "head": e["source"] in owner_ids,
                "tail": e["target"] in owner_ids,
            },
        }

    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for e in edges:
        s = assign[id(e)]
        buckets[s].append(record(e, s))

    files = {"train": "train_facts.json", "validation": "validation_facts.json",
             "test": "test_facts.json"}
    for split, facts in buckets.items():
        rel = Counter(f["relation"] for f in facts)
        owner_touch = sum(1 for f in facts
                          if f["has_internal_graph"]["head"] or f["has_internal_graph"]["tail"])
        payload = {
            "metadata": {
                "split": split,
                "total_facts": len(facts),
                "unique_undirected_edges": len(facts),
                "note": "one record per PrimeKG edge; mirror direction collapsed. "
                        "Add inverse edges in the model, not in this file.",
                "facts_touching_an_internal_graph": owner_touch,
                "relation_counts": dict(rel.most_common()),
                "seed": args.seed,
                "source": "root_graph.json (GoG subset)",
            },
            "facts": facts,
        }
        (out_dir / files[split]).write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------- verify
    keys = {s: {(f["head"], f["relation"], f["tail"]) for f in b} for s, b in buckets.items()}
    undir = {s: {(*sorted((f["head"], f["tail"])), f["relation"]) for f in b}
             for s, b in buckets.items()}
    leak = {
        "test_in_train": len(keys["test"] & keys["train"]),
        "val_in_train": len(keys["validation"] & keys["train"]),
        "test_undirected_in_train": len(undir["test"] & undir["train"]),
        "val_undirected_in_train": len(undir["validation"] & undir["train"]),
        "test_in_val": len(keys["test"] & keys["validation"]),
    }

    print("\n" + "=" * 70)
    print(f"{'split':14s}{'facts':>10s}{'touching internal graph':>26s}")
    for s in ("train", "validation", "test"):
        b = buckets[s]
        touch = sum(1 for f in b if f["has_internal_graph"]["head"] or f["has_internal_graph"]["tail"])
        print(f"{s:14s}{len(b):>10,}{touch:>19,} ({touch/max(1,len(b)):5.1%})")
    print("=" * 70)
    print("leakage:", leak)
    if any(leak.values()):
        print("LEAKAGE DETECTED — do not use these splits", file=sys.stderr)
        return 2
    print("clean: no held-out edge appears in train in either direction")

    print("\nper-relation held-out counts:")
    all_rels = sorted({f["relation"] for b in buckets.values() for f in b})
    print(f"{'relation':30s}{'train':>9s}{'val':>7s}{'test':>7s}")
    for rel in all_rels:
        c = {s: sum(1 for f in buckets[s] if f["relation"] == rel) for s in buckets}
        print(f"{rel:30s}{c['train']:>9,}{c['validation']:>7,}{c['test']:>7,}")
    if held_back:
        print(f"\nkept wholly in train (under --min-held-out): {', '.join(held_back)}")

    print(f"\nwrote {out_dir / files['train']}, {files['validation']}, {files['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
