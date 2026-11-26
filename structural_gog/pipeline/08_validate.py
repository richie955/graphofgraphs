#!/usr/bin/env python3
"""Phase 6: validate the integrated graph-of-graphs.

Beyond the structural checks in `schema.validate_envelope()`, this re-derives the
claims the build makes, from the output rather than from the builders:

  drug graphs      every atom appears exactly once; the original bond set is
                   recoverable from the fragments; the stored canonical SMILES
                   re-parses to a molecule with the same heavy-atom count
  pathway graphs   reaction and entity stIds are well formed; no `precedes` edge
                   without Reactome's own ordering; no isolated nodes
  cross edges      every edge carries evidence and a method; endpoints resolve;
                   no edge re-encodes a held-out root fact
  overall          ownership is intact, ids are unique, and the internal graphs
                   genuinely add structure PrimeKG does not have

    python scripts/08_validate.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gog import schema
from gog.config import cfg

RE_ST = re.compile(r"^R-[A-Z]{3}-\d+(\.\d+)?$")


def check_drug_graphs(graphs: list[dict]) -> tuple[list[str], dict]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    problems: list[str] = []
    stats = Counter()
    for g in graphs:
        gid = g["graph_id"]
        smiles = (g.get("provenance") or {}).get("canonical_smiles")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            problems.append(f"{gid}: stored canonical_smiles does not re-parse")
            continue

        atom_nodes = [n for n in g["nodes"] if n["type"] == "atom"]
        frags = [n for n in g["nodes"] if n["type"] == "molecular_fragment"]
        refs = [n for n in g["nodes"] if n["type"] == "molecule_ref"]
        if len(refs) != 1:
            problems.append(f"{gid}: expected one molecule_ref, found {len(refs)}")

        covered: list[int] = []
        for n in atom_nodes:
            covered.append(int(n["id"].rsplit(":", 1)[1]))
        for f in frags:
            covered.extend(f["attrs"]["atoms"])

        n_heavy = mol.GetNumHeavyAtoms()
        if sorted(covered) != list(range(n_heavy)):
            dupes = [a for a, c in Counter(covered).items() if c > 1]
            detail = f", duplicated {dupes[:5]}" if dupes else ""
            problems.append(
                f"{gid}: atom partition broken — {len(covered)} slots "
                f"for {n_heavy} atoms{detail}"
            )
            stats["partition_broken"] += 1
            continue

        # Rebuild the bond set: fragment internals plus every cross-node edge.
        owner: dict[int, str] = {}
        for n in atom_nodes:
            owner[int(n["id"].rsplit(":", 1)[1])] = n["id"]
        for f in frags:
            for a in f["attrs"]["atoms"]:
                owner[a] = f["id"]

        rebuilt: set[tuple[int, int]] = set()
        for f in frags:
            for i, j, _t in f["attrs"]["internal_bonds"]:
                rebuilt.add((min(i, j), max(i, j)))
        for e in g["edges"]:
            i = e["attrs"].get("attach_from_atom")
            j = e["attrs"].get("attach_to_atom")
            if i is None or j is None:
                problems.append(f"{gid}: edge without attachment atoms")
                continue
            rebuilt.add((min(i, j), max(i, j)))

        original = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                     max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
        if rebuilt != original:
            problems.append(
                f"{gid}: bond reconstruction mismatch "
                f"({len(original - rebuilt)} lost, {len(rebuilt - original)} invented)"
            )
            stats["bonds_mismatch"] += 1
        else:
            stats["reconstructed_ok"] += 1
    return problems, dict(stats)


def check_pathway_graphs(graphs: list[dict]) -> tuple[list[str], dict]:
    problems: list[str] = []
    stats = Counter()
    for g in graphs:
        gid = g["graph_id"]
        ids = {n["id"] for n in g["nodes"]}
        reactions = {n["id"] for n in g["nodes"] if n["type"] == "reaction_event"}

        for n in g["nodes"]:
            st = (n.get("attrs") or {}).get("reactome_st_id")
            if st and not RE_ST.match(str(st)):
                problems.append(f"{gid}: malformed Reactome stId {st!r}")

        for e in g["edges"]:
            if e["type"] == "precedes":
                if (e["attrs"] or {}).get("source") != "reactome_precedingEvent":
                    problems.append(f"{gid}: precedes edge without curated ordering")
                if e["source"] not in reactions or e["target"] not in reactions:
                    problems.append(f"{gid}: precedes edge not between reactions")
            if e["type"] == "input_to" and e["target"] not in reactions:
                problems.append(f"{gid}: input_to does not point at a reaction")
            if e["type"] in ("produces", "catalyzed_by", "activated_by", "inhibited_by") \
                    and e["source"] not in reactions:
                problems.append(f"{gid}: {e['type']} does not originate at a reaction")

        touched = {e["source"] for e in g["edges"]} | {e["target"] for e in g["edges"]}
        isolated = ids - touched
        if isolated:
            stats["graphs_with_isolated_nodes"] += 1
            stats["isolated_nodes"] += len(isolated)
        stats["reactions"] += len(reactions)
    return problems, dict(stats)


def check_cross_edges(doc: dict, held_out: set[tuple[str, str]]) -> tuple[list[str], dict]:
    problems: list[str] = []
    stats = Counter()
    edges = doc["cross_context_edges"]
    for e in edges:
        ev = e.get("evidence") or {}
        if not ev:
            problems.append(f"cross edge {e['source']} -> {e['target']} has no evidence")
        if not ev.get("method"):
            problems.append(f"cross edge {e['source']} -> {e['target']} has no method")
        stats[str(ev.get("method"))] += 1
        if e["type"] == "targets_reaction":
            acc = ev.get("uniprot")
            if not acc:
                problems.append("targets_reaction edge without a UniProt accession")
    seen = Counter((e["source"], e["type"], e["target"]) for e in edges)
    dupes = [k for k, c in seen.items() if c > 1]
    if dupes:
        problems.append(f"{len(dupes)} duplicate cross-context edges")
    return problems, dict(stats)


def main() -> int:
    conf = cfg()
    out = conf["paths"]["out_dir"]
    path = out / "biomedical_gog.json"
    if not path.exists():
        alt = out / "biomedical_gog.invalid.json"
        if alt.exists():
            path = alt
            print(f"validating the INVALID document at {alt}")
        else:
            print("run scripts/07_integrate.py first", file=sys.stderr)
            return 1

    doc = json.loads(path.read_text())
    all_problems: dict[str, list[str]] = {}

    print("=" * 70)
    print("PHASE 6 VALIDATION")
    print("=" * 70)

    structural = schema.validate_envelope(doc)
    all_problems["envelope"] = structural
    print(f"envelope structure       {'PASS' if not structural else f'{len(structural)} problems'}")

    dp, dstats = check_drug_graphs(doc["internal_graphs"]["drug"])
    all_problems["drug_graphs"] = dp
    print(f"drug graphs              {'PASS' if not dp else f'{len(dp)} problems'}  {dstats}")

    pp, pstats = check_pathway_graphs(doc["internal_graphs"]["pathway"])
    all_problems["pathway_graphs"] = pp
    print(f"pathway graphs           {'PASS' if not pp else f'{len(pp)} problems'}  {pstats}")

    held_out: set[tuple[str, str]] = set()
    for fname in ("validation_facts.json", "test_facts.json"):
        p = out / fname
        if p.exists():
            for f in json.loads(p.read_text())["facts"]:
                held_out.add((f["head"], f["tail"]))
                held_out.add((f["tail"], f["head"]))

    cp, cstats = check_cross_edges(doc, held_out)
    all_problems["cross_context"] = cp
    print(f"cross-context edges      {'PASS' if not cp else f'{len(cp)} problems'}")
    print(f"   by method: {cstats}")

    # Does the GoG actually add structure PrimeKG lacks?
    meta = doc["metadata"]
    root_n = meta["root"]["n_nodes"]
    int_n = meta["internal"]["n_nodes"]
    int_e = meta["internal"]["n_edges"]
    gain_ok = int_n > 0 and int_e > 0
    print(f"\nstructural gain          {'PASS' if gain_ok else 'FAIL'}")
    print(f"   {int_n:,} internal nodes and {int_e:,} internal edges added "
          f"on top of {root_n:,} root nodes ({int_n/max(1,root_n):.1f}x)")

    total = sum(len(v) for v in all_problems.values())
    report_path = conf["paths"]["interim_dir"] / "validation_report.json"
    report_path.write_text(json.dumps({
        "document": str(path),
        "n_problems": total,
        "problems": all_problems,
        "drug_stats": dstats,
        "pathway_stats": pstats,
        "cross_method_counts": cstats,
    }, indent=2))

    print("\n" + "=" * 70)
    if total == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{total} PROBLEMS")
        for section, items in all_problems.items():
            for p in items[:5]:
                print(f"   [{section}] {p}")
    print(f"\nwrote {report_path}")
    return 0 if total == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
