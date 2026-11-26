#!/usr/bin/env python3
"""Audit the flat train/val/test fact files before building anything on them.

Checks, in order of how badly each one hurts:
  1. Are source identifiers present? Without them no internal graph can attach.
  2. Cross-split leakage: exact, reverse-direction, and undirected.
  3. Within-split direction duplication (does one edge get scored twice?).
  4. Relation balance: is the headline metric really one relation?
  5. Owner-node reach: can the drug and pathway graphs affect the metric at all?

Writes data/interim/split_audit.json.

    python scripts/00_inspect_splits.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog import splits as sp
from gog.config import cfg


def main() -> int:
    conf = cfg()
    try:
        meta, data = sp.load_all()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("=" * 72)
    print("SPLIT FILES")
    print("=" * 72)
    for name, m in meta.items():
        print(f"{name:11s} {m}")

    a = sp.audit(data)

    print("\n" + "=" * 72)
    print("SIZE AND DIRECTION DUPLICATION")
    print("=" * 72)
    print(f"{'split':12s}{'facts':>12s}{'unique edges':>15s}{'factor':>9s}{'in GoG universe':>18s}")
    for name in data:
        print(f"{name:12s}{a.counts[name]:>12,}{a.unique_edge_counts[name]:>15,}"
              f"{a.duplication_factor[name]:>9.2f}{a.gog_universe[name]:>18,}")

    print("\n" + "=" * 72)
    print("LEAKAGE AGAINST TRAIN")
    print("=" * 72)
    for name in a.cross_split_exact:
        print(f"{name:12s} exact {a.cross_split_exact[name]:>7,}   "
              f"reverse {a.cross_split_reverse[name]:>7,}   "
              f"undirected {a.cross_split_undirected[name]:>7,}")
    if not any(a.cross_split_undirected.values()):
        print("  clean: no held-out edge appears in train in either direction")

    print("\n" + "=" * 72)
    print("RELATION BALANCE")
    print("=" * 72)
    for name in data:
        print(f"\n{name}:")
        for rel, c, share in a.relation_share[name][:6]:
            bar = "#" * int(share * 50)
            print(f"   {rel:26s}{c:>8,}  {share:6.1%} {bar}")

    print("\n" + "=" * 72)
    print("UNIQUE EDGES PER CANDIDATE TARGET RELATION")
    print("=" * 72)
    names = list(data)
    print(f"{'relation':24s}" + "".join(f"{n:>14s}" for n in names))
    for rel, per in a.target_relation_edges.items():
        print(f"{rel:24s}" + "".join(f"{per.get(n, 0):>14,}" for n in names))

    print("\n" + "=" * 72)
    print("OWNER-NODE REACH (can the internal graphs move the metric?)")
    print("=" * 72)
    for name, cov in a.owner_coverage.items():
        print(f"{name:12s} drug-touching {cov['facts_touching_drug']:>8,} "
              f"({cov['drug_share']:6.2%})   pathway-touching "
              f"{cov['facts_touching_pathway']:>6,} ({cov['pathway_share']:6.2%})")

    print("\n" + "=" * 72)
    print(f"FINDINGS ({len(a.problems)})")
    print("=" * 72)
    if not a.problems:
        print("none")
    for i, p in enumerate(a.problems, 1):
        print(f"{i:2d}. {p}")

    out = conf["paths"]["interim_dir"] / "split_audit.json"
    out.write_text(json.dumps({"metadata": meta, "audit": asdict(a)}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
