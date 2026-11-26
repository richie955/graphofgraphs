"""Phase 4: cross-context edges between the drug and pathway levels.

Four edge types, each carrying the evidence that produced it:

    participates_in_reaction  root drug  -> reaction_event
        The drug itself is a curated participant of the reaction. Matched on
        Guide to Pharmacology ligand id (how Reactome actually stores drugs) or
        on ChEBI for drugs that are also ordinary metabolites.

    same_chemical_as          drug ::mol -> molecular_state
        The whole-molecule reference and the Reactome state are the same chemical.

    targets_reaction          root drug  -> reaction_event
        The drug has a PrimeKG protein target, and that protein participates in
        the reaction. Weaker than direct participation, so it is a separate type
        and never merged into the one above.

    same_entity_as            root protein -> molecular_state
        Grounds the pathway level in the root graph. Without these, the pathway
        internal graphs are an island that message passing cannot reach.

Leakage control
---------------
`drop_leaky_edges()` removes cross edges that would hand the model a held-out
root edge. A `targets_reaction` edge built from the PrimeKG target `(drug, P)`
re-encodes that exact fact; if `(drug, P)` is in validation or test, the edge is
a two-hop copy of the answer. The same applies when a cross edge pair creates a
short path between the endpoints of any held-out edge. Both are dropped and
counted, because an unreported leak is worse than a smaller graph.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .reactome import normalise_drug_name
from .schema import CrossContextEdge, drug_mol_ref_id, owner_of, root_id


@dataclass
class DrugKeys:
    """Everything that can identify one drug on the pathway side."""
    drugbank_id: str
    name: str = ""
    chebi: set[str] = field(default_factory=set)        # bare accessions: "15365"
    gtp: set[str] = field(default_factory=set)          # iuphar.ligand ids
    name_key: str = ""


def drug_keys_from_structures(
    structures: dict[str, dict],
    names: dict[str, str] | None = None,
) -> dict[str, DrugKeys]:
    names = names or {}
    out: dict[str, DrugKeys] = {}
    for db_id, rec in structures.items():
        if not rec.get("smiles"):
            continue
        chebi = {str(c).upper().replace("CHEBI:", "") for c in (rec.get("chebi_ids") or [])}
        name = names.get(db_id, "") or ""
        out[db_id] = DrugKeys(
            drugbank_id=db_id,
            name=name,
            chebi={c for c in chebi if c},
            name_key=normalise_drug_name(name) if name else "",
        )
    return out


# --------------------------------------------------------------------------- #
# Indexing the pathway side
# --------------------------------------------------------------------------- #


@dataclass
class PathwayIndex:
    """Reverse indexes over every pathway internal graph's entity nodes."""
    by_chebi: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_gtp: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_uniprot: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_name_key: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Name index restricted to entities Reactome already types as therapeutics.
    # Matching a drug name against *any* entity would let a metabolite called
    # the same thing produce a false participation edge; restricting it to
    # ReferenceTherapeutic entities keeps the match drug-to-drug.
    by_therapeutic_name: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    entity_reactions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    entity_roles: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    node_name: dict[str, str] = field(default_factory=dict)
    node_type: dict[str, str] = field(default_factory=dict)


ROLE_OF_EDGE = {
    "input_to": "input",
    "produces": "output",
    "catalyzed_by": "catalyst",
    "activated_by": "activator",
    "inhibited_by": "inhibitor",
}


def index_pathways(graphs: Iterable[dict]) -> PathwayIndex:
    idx = PathwayIndex()
    for g in graphs:
        for n in g["nodes"]:
            nid, ntype = n["id"], n["type"]
            idx.node_name[nid] = n.get("name") or ""
            idx.node_type[nid] = ntype
            if ntype == "reaction_event":
                continue
            attrs = n.get("attrs") or {}
            identifiers = attrs.get("identifiers") or {}
            is_therapeutic = "iuphar.ligand" in identifiers
            for db, idents in identifiers.items():
                for ident in idents:
                    key = str(ident)
                    if db == "chebi":
                        idx.by_chebi[key.upper().replace("CHEBI:", "")].add(nid)
                    elif db == "iuphar.ligand":
                        idx.by_gtp[key].add(nid)
                    elif db == "uniprot":
                        idx.by_uniprot[key].add(nid)
            raw = (n.get("name") or "").split("[")[0].strip()
            if raw:
                name_key = normalise_drug_name(raw)
                idx.by_name_key[name_key].add(nid)
                if is_therapeutic:
                    idx.by_therapeutic_name[name_key].add(nid)

        for e in g["edges"]:
            role = ROLE_OF_EDGE.get(e["type"])
            if role is None:
                continue
            # input_to points state -> reaction; the others point reaction -> state.
            entity, reaction = (e["source"], e["target"]) if e["type"] == "input_to" \
                else (e["target"], e["source"])
            idx.entity_reactions[entity].add(reaction)
            idx.entity_roles[(entity, reaction)].add(role)
    return idx


# --------------------------------------------------------------------------- #
# Edge construction
# --------------------------------------------------------------------------- #


def build_cross_edges(
    drug_keys: dict[str, DrugKeys],
    pathway_graphs: list[dict],
    *,
    protein_targets: dict[str, set[str]] | None = None,
    protein_root_ids: dict[str, str] | None = None,
    allow_name_match: bool = True,
) -> tuple[list[CrossContextEdge], dict[str, Any]]:
    """Returns (edges, stats).

    `protein_targets` maps a DrugBank id to UniProt accessions of its PrimeKG
    targets; `protein_root_ids` maps a UniProt accession to a root protein node id.

    `allow_name_match` adds matches on normalised drug name, against Reactome
    entities that are already typed as therapeutics. It is on because UniChem's
    Guide to Pharmacology coverage is incomplete — abacavir is curated as a
    therapeutic in Reactome but UniChem gives it no gtopdb id — so an
    accession-only policy silently drops real participation edges. Each edge
    records its method, so a name-matched edge can be filtered out later.
    """
    idx = index_pathways(pathway_graphs)
    edges: list[CrossContextEdge] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    def emit(source: str, target: str, etype: str, evidence: dict[str, Any]) -> None:
        key = (source, etype, target)
        if key in seen:
            return
        seen.add(key)
        edges.append(CrossContextEdge(source=source, target=target, type=etype,
                                      evidence=evidence))
        stats[etype] += 1

    # ---- direct participation: the drug IS a reaction participant -----------
    for db_id, keys in drug_keys.items():
        drug_root = root_id("drug", db_id)
        mol_ref = drug_mol_ref_id(db_id)
        hits: list[tuple[str, str, str]] = []           # (entity_node, method, identifier)
        for gtp in keys.gtp:
            for nid in idx.by_gtp.get(gtp, ()):
                hits.append((nid, "guide_to_pharmacology", f"iuphar.ligand:{gtp}"))
        for ch in keys.chebi:
            for nid in idx.by_chebi.get(ch, ()):
                hits.append((nid, "chebi", f"CHEBI:{ch}"))
        if allow_name_match and keys.name_key:
            for nid in idx.by_therapeutic_name.get(keys.name_key, ()):
                hits.append((nid, "therapeutic_name_match", keys.name_key))

        for entity, method, ident in hits:
            emit(mol_ref, entity, "same_chemical_as", {
                "method": method, "participant_id": ident,
                "drugbank_id": db_id,
                "pathway_graph": owner_of(entity),
                "entity_name": idx.node_name.get(entity),
            })
            for reaction in idx.entity_reactions.get(entity, ()):
                roles = sorted(idx.entity_roles.get((entity, reaction), ()))
                emit(drug_root, reaction, "participates_in_reaction", {
                    "method": method, "participant_id": ident,
                    "participant_node": entity, "roles": roles,
                    "reactome_reaction": idx.node_name.get(reaction),
                    "pathway_graph": owner_of(reaction),
                })

    # ---- protein-mediated: the drug's target participates -------------------
    if protein_targets:
        for db_id, accessions in protein_targets.items():
            if db_id not in drug_keys:
                continue
            drug_root = root_id("drug", db_id)
            for acc in accessions:
                for entity in idx.by_uniprot.get(acc, ()):
                    for reaction in idx.entity_reactions.get(entity, ()):
                        roles = sorted(idx.entity_roles.get((entity, reaction), ()))
                        emit(drug_root, reaction, "targets_reaction", {
                            "method": "primekg_target_uniprot",
                            "uniprot": acc,
                            "participant_node": entity,
                            "roles": roles,
                            "reactome_reaction": idx.node_name.get(reaction),
                            "pathway_graph": owner_of(reaction),
                        })

    # ---- grounding: pathway proteins back onto root protein nodes -----------
    if protein_root_ids:
        for acc, root_node in protein_root_ids.items():
            for entity in idx.by_uniprot.get(acc, ()):
                emit(root_node, entity, "same_entity_as", {
                    "method": "uniprot_accession",
                    "uniprot": acc,
                    "entity_name": idx.node_name.get(entity),
                    "pathway_graph": owner_of(entity),
                })

    summary = {
        "edge_type_counts": dict(stats),
        "n_edges": len(edges),
        "drugs_with_direct": len({e.source for e in edges
                                  if e.type == "participates_in_reaction"}),
        "drugs_with_mediated": len({e.source for e in edges
                                    if e.type == "targets_reaction"}),
        "grounded_proteins": len({e.source for e in edges if e.type == "same_entity_as"}),
        "name_matching_enabled": allow_name_match,
    }
    return edges, summary


# --------------------------------------------------------------------------- #
# Leakage control
# --------------------------------------------------------------------------- #


def drop_leaky_edges(
    edges: list[CrossContextEdge],
    held_out_pairs: set[tuple[str, str]],
    *,
    protein_root_of_uniprot: dict[str, str] | None = None,
) -> tuple[list[CrossContextEdge], dict[str, Any]]:
    """Remove cross edges that re-encode a held-out root edge.

    Two patterns are caught:

    1. A `targets_reaction` edge whose `(drug, target protein)` pair is itself
       held out. The edge exists *because* of that fact, so keeping it leaks the
       label directly.
    2. A drug and a protein that are joined by a held-out root edge and also both
       attached to the same reaction node. Those two cross edges together form a
       two-hop path between the held-out endpoints.
    """
    protein_root_of_uniprot = protein_root_of_uniprot or {}
    kept: list[CrossContextEdge] = []
    dropped_direct = 0
    dropped_path = 0

    # Reaction -> root protein nodes implicated at that reaction.
    protein_at_reaction: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.type != "targets_reaction":
            continue
        acc = e.evidence.get("uniprot")
        root_protein = protein_root_of_uniprot.get(str(acc))
        if root_protein:
            protein_at_reaction[e.target].add(root_protein)

    for e in edges:
        if e.type == "targets_reaction":
            acc = str(e.evidence.get("uniprot") or "")
            root_protein = protein_root_of_uniprot.get(acc)
            if root_protein and (e.source, root_protein) in held_out_pairs:
                dropped_direct += 1
                continue
            leaks = False
            for other in protein_at_reaction.get(e.target, ()):
                if (e.source, other) in held_out_pairs:
                    leaks = True
                    break
            if leaks:
                dropped_path += 1
                continue
        kept.append(e)

    return kept, {
        "dropped_reencoding_held_out_edge": dropped_direct,
        "dropped_two_hop_path": dropped_path,
        "kept": len(kept),
        "removed_total": len(edges) - len(kept),
    }
