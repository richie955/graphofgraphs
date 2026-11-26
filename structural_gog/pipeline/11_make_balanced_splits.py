#!/usr/bin/env python3
"""Rebalance the hybrid edge set into a true 80 / 10 / 20 split.

`10_make_hybrid_splits.py` only grew the training side, leaving 96.6 / 1.7 / 1.7.
This repartitions the *whole* hybrid universe at 8:1:1, stratified per relation,
and then proves the result is leak-free rather than asserting it.

Invariants enforced:

* **One record per edge.** PrimeKG stores both directions; both belong to the same
  split and only one is emitted, so no metric counts an edge twice.
* **Stratified per relation**, so a rare relation keeps a proportional held-out
  share instead of vanishing into the training set.
* **No cold-start endpoints.** A held-out edge whose node appears nowhere in train
  is moved to train: an unseen node has no embedding to rank with.
* **Verified, not assumed.** Every split pair is checked for exact, reverse and
  undirected overlap, and the exit code is non-zero if any check fails.

Because the held-out sets change, the cross-context leakage filter has to run
again afterwards — `06_build_cross_edges.py` reads these files. The script says so
at the end.

    python scripts/11_make_balanced_splits.py
    python scripts/11_make_balanced_splits.py --val-frac 0.1 --test-frac 0.1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog.config import cfg


def undirected(f: dict) -> tuple[str, str, str]:
    a, b = sorted((f["head"], f["tail"]))
    return (a, f["relation"], b)


def directed(f: dict) -> tuple[str, str, str]:
    return (f["head"], f["relation"], f["tail"])


def reverse(f: dict) -> tuple[str, str, str]:
    return (f["tail"], f["relation"], f["head"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-held-out", type=int, default=10,
                    help="relations with fewer unique edges stay wholly in train")
    args = ap.parse_args()

    conf = cfg()
    out, interim = conf["paths"]["out_dir"], conf["paths"]["interim_dir"]

    sources = ["hybrid_train_facts.json", "validation_facts.json", "test_facts.json"]
    missing = [s for s in sources if not (out / s).exists()]
    if missing:
        print(f"missing {', '.join(missing)}; run scripts/10_make_hybrid_splits.py first",
              file=sys.stderr)
        return 1

    # ---- pool every edge, collapsing duplicates ---------------------------
    pool: dict[tuple[str, str, str], dict] = {}
    for name in sources:
        for f in json.loads((out / name).read_text())["facts"]:
            pool.setdefault(undirected(f), f)
    edges = list(pool.values())
    print(f"hybrid universe: {len(edges):,} unique undirected edges")

    owner_ids: set[str] = set()
    seeds_path = interim / "seeds.json"
    if seeds_path.exists():
        seeds = json.loads(seeds_path.read_text())
        owner_ids = {f"root:drug:{d}" for d in seeds.get("drugs", [])} | \
                    {f"root:pathway:{p}" for p in seeds.get("pathways", [])}

    def touches_owner(f: dict) -> bool:
        return f["head"] in owner_ids or f["tail"] in owner_ids

    # ---- stratified assignment --------------------------------------------
    rng = random.Random(args.seed)
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for f in edges:
        by_rel[f["relation"]].append(f)

    assign: dict[tuple[str, str, str], str] = {}
    held_back: list[str] = []
    for rel, group in sorted(by_rel.items()):
        group = sorted(group, key=lambda f: (f["head"], f["tail"]))
        rng.shuffle(group)
        if len(group) < args.min_held_out:
            held_back.append(f"{rel}({len(group)})")
            for f in group:
                assign[undirected(f)] = "train"
            continue
        n_test = max(1, int(round(len(group) * args.test_frac)))
        n_val = max(1, int(round(len(group) * args.val_frac)))
        for i, f in enumerate(group):
            key = undirected(f)
            assign[key] = ("test" if i < n_test
                           else "validation" if i < n_test + n_val
                           else "train")

    # ---- no cold-start endpoints ------------------------------------------
    train_nodes: set[str] = set()
    for f in edges:
        if assign[undirected(f)] == "train":
            train_nodes.add(f["head"])
            train_nodes.add(f["tail"])
    moved = 0
    for f in edges:
        key = undirected(f)
        if assign[key] == "train":
            continue
        if f["head"] not in train_nodes or f["tail"] not in train_nodes:
            assign[key] = "train"
            train_nodes.add(f["head"])
            train_nodes.add(f["tail"])
            moved += 1
    if moved:
        print(f"moved {moved:,} held-out edges to train (endpoint unseen in train)")

    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for f in edges:
        split = assign[undirected(f)]
        rec = dict(f)
        rec["split"] = split
        rec["touches_internal_graph"] = touches_owner(f)
        buckets[split].append(rec)

    total = len(edges)
    print(f"\nsplit: train {len(buckets['train']):,} "
          f"({len(buckets['train'])/total:.1%}), "
          f"validation {len(buckets['validation']):,} "
          f"({len(buckets['validation'])/total:.1%}), "
          f"test {len(buckets['test']):,} ({len(buckets['test'])/total:.1%})")

    # ---- verification ------------------------------------------------------
    print("\n" + "=" * 72)
    print("LEAKAGE VERIFICATION")
    print("=" * 72)
    dkeys = {s: {directed(f) for f in b} for s, b in buckets.items()}
    rkeys = {s: {reverse(f) for f in b} for s, b in buckets.items()}
    ukeys = {s: {undirected(f) for f in b} for s, b in buckets.items()}

    checks: list[tuple[str, int]] = []
    for a, b in (("test", "train"), ("validation", "train"), ("test", "validation")):
        checks.append((f"{a} exact in {b}", len(dkeys[a] & dkeys[b])))
        checks.append((f"{a} reversed in {b}", len(rkeys[a] & dkeys[b])))
        checks.append((f"{a} undirected in {b}", len(ukeys[a] & ukeys[b])))

    dup_within = [(s, len(b) - len(ukeys[s])) for s, b in buckets.items()]
    for s, n in dup_within:
        checks.append((f"{s} internal duplicate edges", n))

    cold = 0
    for s in ("validation", "test"):
        for f in buckets[s]:
            if f["head"] not in train_nodes or f["tail"] not in train_nodes:
                cold += 1
    checks.append(("held-out endpoints unseen in train", cold))

    failed = [(label, n) for label, n in checks if n != 0]
    for label, n in checks:
        print(f"   {'FAIL' if n else 'pass'}  {label:46s} {n}")

    # ---- per-relation table ------------------------------------------------
    print("\n" + "=" * 72)
    print("PER-RELATION DISTRIBUTION")
    print("=" * 72)
    print(f"{'relation':30s}{'train':>10s}{'val':>9s}{'test':>9s}{'test %':>9s}")
    rel_table: dict[str, dict[str, int]] = {}
    for rel in sorted(by_rel):
        c = {s: sum(1 for f in buckets[s] if f["relation"] == rel) for s in buckets}
        rel_table[rel] = c
        tot = sum(c.values())
        print(f"{rel:30s}{c['train']:>10,}{c['validation']:>9,}{c['test']:>9,}"
              f"{c['test']/max(1,tot):>8.1%}")

    # ---- write -------------------------------------------------------------
    files = {"train": "balanced_train_facts.json",
             "validation": "balanced_validation_facts.json",
             "test": "balanced_test_facts.json"}
    for split, facts in buckets.items():
        touch = sum(1 for f in facts if f["touches_internal_graph"])
        payload = {
            "metadata": {
                "split": split,
                "total_facts": len(facts),
                "unique_undirected_edges": len(facts),
                "share_of_universe": round(len(facts) / total, 4),
                "ratio_target": {"train": 0.8, "validation": args.val_frac,
                                 "test": args.test_frac},
                "facts_touching_an_internal_graph": touch,
                "share_touching_an_internal_graph": round(touch / max(1, len(facts)), 4),
                "relation_counts": dict(Counter(f["relation"] for f in facts).most_common()),
                "seed": args.seed,
                "source": "hybrid universe (GoG subset + flat files), repartitioned 8:1:1",
                "note": "one record per PrimeKG edge; the mirror direction is not "
                        "emitted. Add inverse edges in the model, not in this file.",
            },
            "facts": facts,
        }
        (out / files[split]).write_text(json.dumps(payload, indent=2))

    # Evaluation slice: the held-out facts the internal graphs touch directly.
    for split in ("validation", "test"):
        slice_facts = [f for f in buckets[split] if f["touches_internal_graph"]]
        (out / f"balanced_{split}_gog_slice.json").write_text(json.dumps({
            "metadata": {
                "split": split,
                "total_facts": len(slice_facts),
                "parent": files[split],
                "note": "subset of the held-out split whose head or tail owns an "
                        "internal graph; report metrics on this slice alongside the "
                        "full split to isolate the GoG's contribution",
                "relation_counts": dict(Counter(f["relation"] for f in slice_facts).most_common()),
            },
            "facts": slice_facts,
        }, indent=2))

    (interim / "balanced_split_audit.json").write_text(json.dumps({
        "universe_unique_edges": total,
        "counts": {s: len(b) for s, b in buckets.items()},
        "shares": {s: round(len(b) / total, 4) for s, b in buckets.items()},
        "moved_for_cold_start": moved,
        "relations_held_back": held_back,
        "verification": {label: n for label, n in checks},
        "per_relation": rel_table,
    }, indent=2))

    print("\n" + "=" * 72)
    for split in ("train", "validation", "test"):
        b = buckets[split]
        touch = sum(1 for f in b if f["touches_internal_graph"])
        print(f"{split:12s}{len(b):>9,} facts   {len(b)/total:>6.1%}   "
              f"touching an internal graph {touch:>7,} ({touch/max(1,len(b)):.1%})")
    if held_back:
        print(f"\nkept wholly in train (under --min-held-out): {', '.join(held_back)}")
    print("=" * 72)

    if failed:
        print("LEAKAGE CHECK FAILED:", failed, file=sys.stderr)
        return 2
    print("ALL LEAKAGE CHECKS PASSED")
    print("\nNEXT: the held-out sets changed, so re-run the cross-context filter:")
    print("  python scripts/06_build_cross_edges.py --splits balanced")
    print("  python scripts/07_integrate.py && python scripts/08_validate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
