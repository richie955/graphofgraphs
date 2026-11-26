"""The contract between the four workstreams.

Nobody writes an extraction script until these IDs and node/edge types are agreed.
Everything downstream (JSON export, PyG loader, validation) keys off this module.

ID grammar
----------
    root:<type>:<source_id>                 root:drug:DB00945
    drug:<db_id>::atom:<idx>                drug:DB00945::atom:7
    drug:<db_id>::frag:<idx>                drug:DB00945::frag:3
    drug:<db_id>::mol                       drug:DB00945::mol      (whole-molecule ref)
    pathway:<st_id>::rxn:<st_id>            pathway:R-HSA-109582::rxn:R-HSA-109581
    pathway:<st_id>::entity:<st_id>         pathway:R-HSA-109582::entity:R-HSA-114313

`atom:<idx>` uses the RDKit atom index of the *canonical* molecule written by
`canonical_smiles()`, so indices are reproducible across machines. Store that
SMILES on the internal graph or the indices mean nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "0.1.0"

# --------------------------------------------------------------------------- #
# Node / edge vocabularies. Adding a member here is a team decision, not a
# local one: the PyG loader turns each into its own tensor block.
# --------------------------------------------------------------------------- #

RootNodeType = Literal[
    "drug", "disease", "gene/protein", "pathway", "molecular_function",
    "cellular_component", "biological_process", "effect/phenotype",
    "anatomy", "exposure",
]

DrugNodeType = Literal["atom", "molecular_fragment", "molecule_ref"]
DrugEdgeType = Literal["chemical_bond", "chemical_attachment", "fragment_member"]

PathwayNodeType = Literal["reaction_event", "molecular_state", "complex_state", "set_state"]
PathwayEdgeType = Literal[
    "input_to", "produces", "catalyzed_by", "activated_by", "inhibited_by",
    "precedes", "set_member",
]

OwnerLinkType = Literal["owns_internal_graph"]

CrossContextEdgeType = Literal[
    "participates_in_reaction",   # drug owner  -> reaction_event   (drug is a curated participant)
    "targets_reaction",           # drug owner  -> reaction_event   (via a PrimeKG target protein)
    "same_chemical_as",           # drug mol_ref-> molecular_state
    "same_entity_as",             # root protein-> molecular_state  (grounds pathway level in root)
]

BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

# --------------------------------------------------------------------------- #
# ID builders. Use these; never f-string an ID by hand.
# --------------------------------------------------------------------------- #

_RE_DB = re.compile(r"^DB\d{5}$")
_RE_ST = re.compile(r"^R-[A-Z]{3}-\d+(\.\d+)?$")


def root_id(node_type: str, source_id: str) -> str:
    return f"root:{node_type}:{source_id}"


def drug_graph_id(db_id: str) -> str:
    _check(_RE_DB, db_id, "DrugBank id")
    return f"drug:{db_id}"


def drug_atom_id(db_id: str, atom_idx: int) -> str:
    return f"{drug_graph_id(db_id)}::atom:{int(atom_idx)}"


def drug_fragment_id(db_id: str, frag_idx: int) -> str:
    return f"{drug_graph_id(db_id)}::frag:{int(frag_idx)}"


def drug_mol_ref_id(db_id: str) -> str:
    return f"{drug_graph_id(db_id)}::mol"


def pathway_graph_id(st_id: str) -> str:
    _check(_RE_ST, st_id, "Reactome stId")
    return f"pathway:{st_id}"


def pathway_reaction_id(pathway_st_id: str, rxn_st_id: str) -> str:
    _check(_RE_ST, rxn_st_id, "Reactome stId")
    return f"{pathway_graph_id(pathway_st_id)}::rxn:{rxn_st_id}"


def pathway_entity_id(pathway_st_id: str, entity_st_id: str) -> str:
    _check(_RE_ST, entity_st_id, "Reactome stId")
    return f"{pathway_graph_id(pathway_st_id)}::entity:{entity_st_id}"


def owner_of(internal_node_id: str) -> str:
    """'drug:DB00945::atom:7' -> 'drug:DB00945'. Raises on a root id."""
    if "::" not in internal_node_id:
        raise ValueError(f"not an internal node id: {internal_node_id!r}")
    return internal_node_id.split("::", 1)[0]


def _check(rx: re.Pattern[str], value: str, what: str) -> None:
    if not rx.match(value):
        raise ValueError(f"bad {what}: {value!r}")


# --------------------------------------------------------------------------- #
# Record shapes. Keep them flat and JSON-native: `attrs` is a free dict, but
# anything another workstream reads should be promoted to a real field.
# --------------------------------------------------------------------------- #


@dataclass
class Node:
    id: str
    type: str
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalGraph:
    """One drug molecule or one pathway, with the provenance to rebuild it."""
    graph_id: str                      # drug:DB00945  /  pathway:R-HSA-109582
    owner_root_id: str                 # root:drug:DB00945
    kind: Literal["drug", "pathway"]
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    # drug  : {canonical_smiles, pubchem_cid, source, heavy_atoms, fragmenter}
    # pathway: {st_id, species, reactome_release, truncated_from, api_fetched_at}


@dataclass
class CrossContextEdge:
    source: str
    target: str
    type: str
    evidence: dict[str, Any] = field(default_factory=dict)
    # {method: "chebi_participant"|"primekg_target", reactome_reaction: ...,
    #  participant_id: "CHEBI:15365", role: "input", mapped_via: [...]}


def envelope(
    root_nodes: list[Node],
    root_edges: list[Edge],
    drug_graphs: list[InternalGraph],
    pathway_graphs: list[InternalGraph],
    owner_links: list[Edge],
    cross_edges: list[CrossContextEdge],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """The one top-level JSON shape. biomedical_gog.json is exactly this."""
    from dataclasses import asdict

    return {
        "schema_version": SCHEMA_VERSION,
        "root_graph": {
            "nodes": [asdict(n) for n in root_nodes],
            "edges": [asdict(e) for e in root_edges],
        },
        "internal_graphs": {
            "drug": [asdict(g) for g in drug_graphs],
            "pathway": [asdict(g) for g in pathway_graphs],
        },
        "owner_links": [asdict(e) for e in owner_links],
        "cross_context_edges": [asdict(e) for e in cross_edges],
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Validation. Phase 6 calls this; so should every builder before it writes.
# --------------------------------------------------------------------------- #


def validate_envelope(doc: dict[str, Any]) -> list[str]:
    """Return a list of problems. Empty list means the document is consistent."""
    problems: list[str] = []
    root_ids = {n["id"] for n in doc["root_graph"]["nodes"]}
    if len(root_ids) != len(doc["root_graph"]["nodes"]):
        problems.append("duplicate root node ids")

    internal_ids: set[str] = set()
    graph_ids: set[str] = set()
    for kind in ("drug", "pathway"):
        for g in doc["internal_graphs"][kind]:
            if g["graph_id"] in graph_ids:
                problems.append(f"duplicate internal graph {g['graph_id']}")
            graph_ids.add(g["graph_id"])
            if g["owner_root_id"] not in root_ids:
                problems.append(f"{g['graph_id']} owner {g['owner_root_id']} not in root graph")
            local = {n["id"] for n in g["nodes"]}
            if len(local) != len(g["nodes"]):
                problems.append(f"duplicate node ids inside {g['graph_id']}")
            for n in g["nodes"]:
                if not n["id"].startswith(g["graph_id"] + "::"):
                    problems.append(f"node {n['id']} does not belong to {g['graph_id']}")
            for e in g["edges"]:
                for endpoint in (e["source"], e["target"]):
                    if endpoint not in local:
                        problems.append(f"{g['graph_id']} edge endpoint {endpoint} missing")
            internal_ids |= local

    all_ids = root_ids | internal_ids
    for e in doc["owner_links"]:
        if e["source"] not in root_ids or e["target"] not in graph_ids:
            problems.append(f"owner link {e['source']} -> {e['target']} unresolved")
    for e in doc["cross_context_edges"]:
        for endpoint in (e["source"], e["target"]):
            if endpoint not in all_ids:
                problems.append(f"cross edge endpoint {endpoint} unresolved")
        if not e.get("evidence"):
            problems.append(f"cross edge {e['source']} -> {e['target']} carries no evidence")
    return problems
