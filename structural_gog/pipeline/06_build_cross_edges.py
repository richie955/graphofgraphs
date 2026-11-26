#!/usr/bin/env python3
"""Phase 4: build cross-context edges between the drug and pathway levels.

Resolves the identifier mappings Phase 4 needs (DrugBank -> Guide to Pharmacology
and ChEBI via UniChem; Entrez -> UniProt for proteins), matches them against the
pathway internal graphs, then drops any edge that would leak a held-out root fact.

    python scripts/06_build_cross_edges.py
    python scripts/06_build_cross_edges.py --no-name-match
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from gog import cross_context as cc
from gog.config import cfg
from gog.identifiers import entrez_to_uniprot, unichem_xrefs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-name-match", action="store_true",
                    help="accession evidence only; drops therapeutic name matches")
    ap.add_argument("--splits", choices=["subset", "balanced"], default="subset",
                    help="which held-out sets the leakage filter protects")
    args = ap.parse_args()

    conf = cfg()
    out, interim = conf["paths"]["out_dir"], conf["paths"]["interim_dir"]

    for needed in ("root_graph.json", "drug_internal_graphs.json", "pathway_internal_graphs.json"):
        if not (out / needed).exists():
            print(f"missing {out / needed}; run phases 1-3 first", file=sys.stderr)
            return 1

    root = json.loads((out / "root_graph.json").read_text())
    pathway_doc = json.loads((out / "pathway_internal_graphs.json").read_text())
    drug_doc = json.loads((out / "drug_internal_graphs.json").read_text())
    structures = json.loads((interim / "drug_structures.json").read_text())

    built_drugs = {g["graph_id"].split(":", 1)[1] for g in drug_doc["graphs"]}
    pathway_graphs = pathway_doc["graphs"]
    print(f"{len(built_drugs)} drug graphs, {len(pathway_graphs)} pathway graphs")

    # ---- root-side lookups -------------------------------------------------
    drug_names = {
        n["attrs"]["source_id"]: n.get("name") or ""
        for n in root["nodes"] if n["type"] == "drug"
    }
    protein_nodes = {
        n["attrs"]["source_id"]: n["id"]
        for n in root["nodes"] if n["type"] == "gene/protein"
    }
    print(f"root graph: {len(drug_names)} drugs, {len(protein_nodes)} proteins")

    # drug -> target gene ids, from the root graph's own drug_protein edges
    targets_by_drug: dict[str, set[str]] = defaultdict(set)
    for e in root["edges"]:
        if "drug_protein" not in e["type"]:
            continue
        for a, b in ((e["source"], e["target"]), (e["target"], e["source"])):
            if a.startswith("root:drug:") and b.startswith("root:gene/protein:"):
                targets_by_drug[a.split(":", 2)[2]].add(b.split(":", 2)[2])
    print(f"{len(targets_by_drug)} drugs have a protein target in the subset")

    # ---- identifier resolution --------------------------------------------
    print("\nresolving DrugBank -> Guide to Pharmacology / ChEBI via UniChem ...")
    keys: dict[str, cc.DrugKeys] = {}
    n_gtp = n_chebi = 0
    for db_id in tqdm(sorted(built_drugs), unit="drug"):
        rec = structures.get(db_id, {})
        x = unichem_xrefs(db_id)
        chebi = {str(c).upper().replace("CHEBI:", "")
                 for c in (rec.get("chebi_ids") or [])}
        if x.chebi:
            chebi.add(x.chebi)
        gtp = {x.gtopdb} if x.gtopdb else set()
        n_gtp += bool(gtp)
        n_chebi += bool(chebi)
        name = drug_names.get(db_id, "")
        keys[db_id] = cc.DrugKeys(
            drugbank_id=db_id, name=name, chebi={c for c in chebi if c}, gtp=gtp,
            name_key=cc.normalise_drug_name(name) if name else "",
        )
    print(f"  {n_gtp} drugs have a Guide to Pharmacology id, {n_chebi} have a ChEBI id")

    print("\nresolving Entrez -> UniProt ...")
    all_genes = sorted(set(protein_nodes))
    gene_to_acc = entrez_to_uniprot(all_genes)
    mapped = {g: a for g, a in gene_to_acc.items() if a}
    print(f"  {len(mapped)}/{len(all_genes)} genes mapped to a reviewed accession")

    acc_to_root: dict[str, str] = {}
    for gid, accs in mapped.items():
        for acc in accs:
            acc_to_root.setdefault(acc, protein_nodes[gid])

    protein_targets: dict[str, set[str]] = {}
    for db_id, gene_ids in targets_by_drug.items():
        accs = {a for g in gene_ids for a in gene_to_acc.get(g, [])}
        if accs:
            protein_targets[db_id] = accs

    # ---- build -------------------------------------------------------------
    edges, summary = cc.build_cross_edges(
        keys, pathway_graphs,
        protein_targets=protein_targets,
        protein_root_ids=acc_to_root,
        allow_name_match=not args.no_name_match,
    )
    print(f"\nbuilt {len(edges):,} cross-context edges: {summary['edge_type_counts']}")

    # ---- leakage filter ----------------------------------------------------
    held_out_files = (
        ("balanced_validation_facts.json", "balanced_test_facts.json")
        if args.splits == "balanced"
        else ("validation_facts.json", "test_facts.json")
    )
    held_out: set[tuple[str, str]] = set()
    for fname in held_out_files:
        p = out / fname
        if not p.exists():
            print(f"  WARNING: {fname} not found; its edges are unprotected",
                  file=sys.stderr)
            continue
        for f in json.loads(p.read_text())["facts"]:
            held_out.add((f["head"], f["tail"]))
            held_out.add((f["tail"], f["head"]))
    print(f"held-out endpoint pairs loaded from {args.splits} splits: {len(held_out):,}")

    kept, leak = cc.drop_leaky_edges(edges, held_out, protein_root_of_uniprot=acc_to_root)
    print(f"leakage filter: {leak}")

    by_method: dict[str, int] = defaultdict(int)
    for e in kept:
        by_method[str(e.evidence.get("method"))] += 1

    payload = {
        "metadata": {
            "n_edges": len(kept),
            "edge_type_counts": {
                t: sum(1 for e in kept if e.type == t)
                for t in ("participates_in_reaction", "same_chemical_as",
                          "targets_reaction", "same_entity_as")
            },
            "method_counts": dict(by_method),
            "leakage_filter": leak,
            "identifier_coverage": {
                "drugs_with_gtopdb": n_gtp,
                "drugs_with_chebi": n_chebi,
                "genes_mapped_to_uniprot": len(mapped),
                "genes_total": len(all_genes),
            },
            "name_matching_enabled": not args.no_name_match,
            "leakage_filter_protects": args.splits,
        },
        "cross_context_edges": [asdict(e) for e in kept],
    }
    (out / "cross_context_edges.json").write_text(json.dumps(payload, indent=2))

    m = payload["metadata"]
    print("\n" + "=" * 68)
    print("CROSS-CONTEXT EDGES")
    print("=" * 68)
    for t, c in m["edge_type_counts"].items():
        print(f"   {t:28s} {c:>8,}")
    print(f"\nby evidence method: {m['method_counts']}")
    print(f"\nwrote {out / 'cross_context_edges.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
