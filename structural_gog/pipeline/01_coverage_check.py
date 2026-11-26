#!/usr/bin/env python3
"""Phase 0: decide whether the GoG is worth building, before building it.

Answers four questions:
  1. How many PrimeKG drugs resolve to a SMILES within the heavy-atom limit?
  2. How many Reactome-human pathways fit under the node limit?
  3. How many direct drug-as-participant cross edges would we get?
  4. How many protein-mediated cross edges would we get?

(3) is the number that decides the project's shape. If it is in the dozens,
the direct matching method cannot carry the cross-context level on its own.

Writes data/interim/coverage_report.json and prints a summary.
Everything it fetches is cached, so re-runs are free and the later phases
reuse the same id maps.

    python scripts/01_coverage_check.py
    python scripts/01_coverage_check.py --drug-sample 0 --pathway-sample 0   # full, slow
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from gog import primekg, pubchem, reactome
from gog.config import cfg


def parse_args() -> argparse.Namespace:
    conf = cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drug-sample", type=int, default=conf["coverage"]["drug_sample"],
                    help="drugs to resolve (0 = all)")
    ap.add_argument("--pathway-sample", type=int, default=conf["coverage"]["pathway_sample"],
                    help="pathways to size (0 = all)")
    ap.add_argument("--seed", type=int, default=conf["coverage"]["seed"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    conf = cfg()
    rng = random.Random(args.seed)
    csv_path = conf["paths"]["primekg_csv"]

    if not Path(csv_path).exists():
        print(f"PrimeKG not found at {csv_path}\n"
              "Download kg.csv from the Harvard Dataverse record "
              "(doi:10.7910/DVN/IXA7BM) and put it there.", file=sys.stderr)
        return 1

    print(f"loading {csv_path} ...")
    df = primekg.load_primekg(csv_path)
    print(f"  {len(df):,} edges, {len(primekg.node_table(df)):,} nodes")

    drugs = primekg.owner_candidates(df, "drug")
    pathways = primekg.owner_candidates(df, "pathway")
    targets = primekg.drug_target_proteins(df)
    print(f"  {len(drugs):,} DrugBank drugs, {len(pathways):,} Reactome human pathways")
    print(f"  {len(targets):,} drugs carry at least one drug_protein edge")

    report: dict = {
        "config": {
            "max_heavy_atoms": conf["subset"]["max_heavy_atoms"],
            "max_pathway_nodes": conf["subset"]["max_pathway_nodes"],
            "drug_sample": args.drug_sample,
            "pathway_sample": args.pathway_sample,
            "seed": args.seed,
        },
        "totals": {
            "primekg_edges": len(df),
            "primekg_nodes": len(primekg.node_table(df)),
            "drug_nodes": len(drugs),
            "pathway_nodes": len(pathways),
        },
    }

    # ---------------------------------------------------------------- drugs
    drug_sample = list(drugs)
    rng.shuffle(drug_sample)
    if args.drug_sample:
        drug_sample = drug_sample[: args.drug_sample]

    max_atoms = conf["subset"]["max_heavy_atoms"]
    structures: dict[str, pubchem.DrugStructure] = {}
    reasons: Counter[str] = Counter()
    chebi_to_drug: dict[str, str] = {}

    print(f"\nresolving {len(drug_sample)} drugs via PubChem (cached; ~5/s cold) ...")
    for d in tqdm(drug_sample, unit="drug"):
        rec = pubchem.resolve(d.source_id)
        structures[d.source_id] = rec
        if not rec.ok():
            reasons["no_smiles"] += 1
        elif rec.heavy_atoms is not None and rec.heavy_atoms > max_atoms:
            reasons["too_large"] += 1
        else:
            reasons["usable"] += 1
        for ch in rec.chebi_ids or []:
            chebi_to_drug.setdefault(ch, d.source_id)

    usable_drugs = [
        db for db, r in structures.items()
        if r.ok() and (r.heavy_atoms is None or r.heavy_atoms <= max_atoms)
    ]
    drug_rate = len(usable_drugs) / max(1, len(drug_sample))
    report["drugs"] = {
        "sampled": len(drug_sample),
        "usable": len(usable_drugs),
        "usable_rate": round(drug_rate, 4),
        "breakdown": dict(reasons),
        "projected_usable_all_drugs": int(round(drug_rate * len(drugs))),
        "with_chebi": sum(1 for r in structures.values() if r.chebi_ids),
    }

    # ------------------------------------------------------------- pathways
    pathway_sample = list(pathways)
    rng.shuffle(pathway_sample)
    if args.pathway_sample:
        pathway_sample = pathway_sample[: args.pathway_sample]

    max_nodes = conf["subset"]["max_pathway_nodes"]
    sizes: list[reactome.PathwaySize] = []
    print(f"\nsizing {len(pathway_sample)} pathways via Reactome ...")
    for p in tqdm(pathway_sample, unit="pathway"):
        try:
            sizes.append(reactome.pathway_size(p.source_id))
        except RuntimeError as exc:
            print(f"  skipped {p.source_id}: {exc}", file=sys.stderr)

    min_nodes = conf["subset"].get("min_pathway_nodes", 1)
    fitting = [s for s in sizes if min_nodes <= s.n_nodes <= max_nodes]
    empty = [s for s in sizes if s.n_nodes == 0]
    too_small = [s for s in sizes if 0 < s.n_nodes < min_nodes]
    node_counts = sorted(s.n_nodes for s in sizes)
    pathway_rate = len(fitting) / max(1, len(sizes))
    report["pathways"] = {
        "sampled": len(sizes),
        "fitting_limit": len(fitting),
        "fitting_rate": round(pathway_rate, 4),
        "min_nodes": min_nodes,
        "too_small": len(too_small),
        "empty_or_unresolved": len(empty),
        "median_nodes": node_counts[len(node_counts) // 2] if node_counts else 0,
        "p90_nodes": node_counts[int(len(node_counts) * 0.9)] if node_counts else 0,
        "max_nodes_seen": node_counts[-1] if node_counts else 0,
        "projected_fitting_all_pathways": int(round(pathway_rate * len(pathways))),
    }

    # -------------------------------------------------- cross-context edges
    # Three candidate matching methods, measured separately:
    #   chebi       drug's ChEBI accession is a pathway participant
    #   therapeutic drug name matches a ReferenceTherapeutic (Guide to Pharmacology)
    #   mediated    a PrimeKG target gene symbol is a pathway participant
    # Reactome curates drugs as ReferenceTherapeutic, so `therapeutic` is where
    # the genuine drug-as-participant edges live; `chebi` only catches drugs that
    # are also ordinary metabolites.
    usable_set = set(usable_drugs)
    chebi_pairs: set[tuple[str, str]] = set()
    therapeutic_pairs: set[tuple[str, str]] = set()
    mediated_pairs: set[tuple[str, str]] = set()

    symbol_to_drugs: dict[str, set[str]] = {}
    for db_id in usable_set:
        for sym in targets.get(db_id, ()):  # gene symbols
            symbol_to_drugs.setdefault(sym, set()).add(db_id)

    name_to_drugs: dict[str, set[str]] = {}
    for d in drug_sample:
        if d.source_id in usable_set:
            key = reactome.normalise_drug_name(d.name)
            if key:
                name_to_drugs.setdefault(key, set()).add(d.source_id)

    for s in sizes:
        for ch in s.chebi:
            db_id = chebi_to_drug.get(ch)
            if db_id in usable_set:
                chebi_pairs.add((db_id, s.st_id))
        for nm in s.therapeutic_names:
            for db_id in name_to_drugs.get(nm, ()):
                therapeutic_pairs.add((db_id, s.st_id))
        for sym in s.gene_names:
            for db_id in symbol_to_drugs.get(sym, ()):
                mediated_pairs.add((db_id, s.st_id))

    direct_pairs = chebi_pairs | therapeutic_pairs
    n_sampled_pathways = max(1, len(sizes))

    # A pair needs BOTH its drug and its pathway to be in the sample, so the
    # projection scales along both axes independently. Scaling only by the
    # pathway ratio (and correcting by the usable-drug rate) understates the
    # result by the drug sampling ratio, which is ~26x at the default sample.
    pathway_scale = len(pathways) / n_sampled_pathways
    usable_sampled = max(1, len(usable_drugs))
    drug_scale = (len(drugs) * drug_rate) / usable_sampled
    scale = pathway_scale * drug_scale
    report["cross_context"] = {
        "note": "pathway-level pairs in the sample; the real builder resolves these "
                "to individual reactions, so reaction-level counts will be higher. "
                "Name matching is an estimate: the real builder should map DrugBank "
                "to Guide to Pharmacology ligand ids and match on accession. "
                "Projections assume matches are spread evenly across drugs and "
                "pathways; in reality they cluster on well-curated pathways, so "
                "treat them as a generous upper bound and confirm on a full run.",
        "pathway_scale": round(pathway_scale, 2),
        "drug_scale": round(drug_scale, 2),
        "chebi_pairs_in_sample": len(chebi_pairs),
        "therapeutic_pairs_in_sample": len(therapeutic_pairs),
        "direct_pairs_in_sample": len(direct_pairs),
        "mediated_pairs_in_sample": len(mediated_pairs),
        "direct_per_pathway": round(len(direct_pairs) / n_sampled_pathways, 3),
        "mediated_per_pathway": round(len(mediated_pairs) / n_sampled_pathways, 3),
        "projected_direct_pairs": int(round(len(direct_pairs) * scale)),
        "projected_mediated_pairs": int(round(len(mediated_pairs) * scale)),
        "drugs_with_any_direct": len({d for d, _ in direct_pairs}),
        "drugs_with_any_mediated": len({d for d, _ in mediated_pairs}),
        "pathways_with_any_therapeutic": sum(1 for s in sizes if s.therapeutics),
        "distinct_therapeutics_seen": len({k for s in sizes for k in s.therapeutics}),
    }

    # ------------------------------------------------------------- persist
    interim = conf["paths"]["interim_dir"]
    (interim / "coverage_report.json").write_text(json.dumps(report, indent=2))
    (interim / "drug_structures.json").write_text(
        json.dumps({k: v.to_dict() for k, v in structures.items()}, indent=2)
    )
    (interim / "pathway_sizes.json").write_text(
        json.dumps(
            [{"st_id": s.st_id, "n_events": s.n_events, "n_entities": s.n_entities,
              "n_nodes": s.n_nodes, "chebi": sorted(s.chebi),
              "gene_names": sorted(s.gene_names), "uniprot": sorted(s.uniprot),
              "therapeutics": s.therapeutics, "schema_classes": s.schema_classes}
             for s in sizes],
            indent=2,
        )
    )

    # -------------------------------------------------------------- summary
    d, p, c = report["drugs"], report["pathways"], report["cross_context"]
    print("\n" + "=" * 68)
    print("COVERAGE SUMMARY")
    print("=" * 68)
    print(f"drugs     {d['usable']}/{d['sampled']} usable ({d['usable_rate']:.1%})"
          f"  -> ~{d['projected_usable_all_drugs']:,} of {len(drugs):,} drugs")
    print(f"          breakdown: {d['breakdown']}")
    print(f"pathways  {p['fitting_limit']}/{p['sampled']} in [{p['min_nodes']}, {max_nodes}] nodes "
          f"({p['fitting_rate']:.1%})  -> ~{p['projected_fitting_all_pathways']:,} "
          f"of {len(pathways):,}")
    print(f"          median {p['median_nodes']} nodes, p90 {p['p90_nodes']}, "
          f"max {p['max_nodes_seen']};  {p['too_small']} under {p['min_nodes']} nodes, "
          f"{p['empty_or_unresolved']} unresolved")
    print(f"cross     chebi       {c['chebi_pairs_in_sample']} pairs in sample")
    print(f"          therapeutic {c['therapeutic_pairs_in_sample']} pairs in sample "
          f"({c['distinct_therapeutics_seen']} distinct Reactome drugs seen, "
          f"{c['pathways_with_any_therapeutic']} pathways carry one)")
    print(f"          direct      {c['direct_pairs_in_sample']} total "
          f"(~{c['projected_direct_pairs']:,} projected)")
    print(f"          mediated    {c['mediated_pairs_in_sample']} pairs in sample "
          f"(~{c['projected_mediated_pairs']:,} projected)")
    print("=" * 68)

    if c["projected_direct_pairs"] < 200:
        print("\nVERDICT: direct participant matching alone is too sparse to move any\n"
              "metric. Build `targets_reaction` (protein-mediated) edges as a first-class\n"
              "edge type, not an extension, and add `same_entity_as` grounding edges.")
    else:
        print("\nVERDICT: direct matching has enough volume to stand on its own.\n"
              "Still add the mediated edges as a separate labelled type for the ablation.")
    if c["therapeutic_pairs_in_sample"] > c["chebi_pairs_in_sample"]:
        print("Note: ReferenceTherapeutic matching beat ChEBI matching, as expected.\n"
              "Phase 4 should map DrugBank -> Guide to Pharmacology, not DrugBank -> ChEBI.")
    print(f"\nwrote {interim / 'coverage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
