"""Phase 3: Reactome pathway -> reaction-level internal graph.

Node types
    reaction_event   a Reaction / BlackBoxEvent / Polymerisation etc.
    molecular_state  SimpleEntity, EntityWithAccessionedSequence, ...
    complex_state    Complex
    set_state        DefinedSet / CandidateSet / OpenSet

Edge types
    input_to      state    -> reaction
    produces      reaction -> state
    catalyzed_by  reaction -> state (the catalyst's physical entity)
    activated_by  reaction -> state (PositiveRegulation)
    inhibited_by  reaction -> state (NegativeRegulation)
    precedes      reaction -> reaction, only from Reactome's `precedingEvent`

Three things the plan's sketch does not say, all verified against the service:

* A `DefinedSet` is a first-class participant, not a convenience wrapper. Sets are
  kept as `set_state` nodes rather than expanded, so the node count stays bounded
  and the "one of these" semantics is not silently turned into "all of these".
* `catalystActivity` entries carry a dbId but no stId, so the catalyst's physical
  entity needs a second lookup per activity.
* Compartment is part of identity: `ATP [cytosol]` and `ATP [mitochondrial matrix]`
  are different stIds, so compartment is stored and never normalised away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import reactome as R
from .schema import (
    Edge, InternalGraph, Node, pathway_entity_id, pathway_graph_id,
    pathway_reaction_id, root_id,
)

SET_CLASSES = {"DefinedSet", "CandidateSet", "OpenSet", "EntitySet"}
COMPLEX_CLASSES = {"Complex"}
EVENT_CLASSES = {
    "Reaction", "BlackBoxEvent", "Polymerisation", "Depolymerisation",
    "FailedReaction", "ReactionLikeEvent",
}


def entity_node_type(schema_class: str) -> str:
    if schema_class in SET_CLASSES:
        return "set_state"
    if schema_class in COMPLEX_CLASSES:
        return "complex_state"
    return "molecular_state"


def _compartments(obj: dict) -> list[str]:
    return [str(c.get("displayName")) for c in (obj.get("compartment") or [])
            if isinstance(c, dict) and c.get("displayName")]


MEMBER_KEYS = ("hasMember", "hasCandidate", "hasComponent", "hasPart", "repeatedUnit")


def _ref_identifiers(obj: dict, *, depth: int = 2,
                     _seen: set[str] | None = None) -> dict[str, list[str]]:
    """chebi / uniprot / iuphar.ligand accessions on an entity, for Phase 4.

    A Complex or a DefinedSet carries no `referenceEntity` of its own; the
    accessions belong to its members, which the service returns unexpanded. So
    members are resolved by stId down to `depth` levels. Without this, every
    complex and set would look identifier-free and the protein-mediated matching
    in Phase 4 would miss every protein that only appears inside a complex.
    """
    out: dict[str, list[str]] = {}
    _seen = _seen if _seen is not None else set()

    def add(ref: dict) -> None:
        db = R.ref_database(ref)
        ident = str(ref.get("identifier") or "")
        if db and ident:
            out.setdefault(db, [])
            if ident not in out[db]:
                out[db].append(ident)

    ref = obj.get("referenceEntity")
    if isinstance(ref, dict):
        add(ref)

    if depth <= 0:
        return out

    for key in MEMBER_KEYS:
        for item in obj.get(key) or []:
            if not isinstance(item, dict):
                continue
            nested = item.get("referenceEntity")
            if isinstance(nested, dict):
                add(nested)
                continue
            st = str(item.get("stId") or item.get("dbId") or "")
            if not st or st in _seen:
                continue
            _seen.add(st)
            full = R.query(st)
            if not full:
                continue
            for db, idents in _ref_identifiers(full, depth=depth - 1, _seen=_seen).items():
                out.setdefault(db, [])
                for i in idents:
                    if i not in out[db]:
                        out[db].append(i)
    return out


@dataclass
class PathwayBuild:
    graph: InternalGraph | None = None
    report: dict[str, Any] = field(default_factory=dict)


def build_pathway_graph(
    st_id: str,
    *,
    max_nodes: int = 200,
    min_nodes: int = 10,
    resolve_catalysts: bool = True,
    prefilter: bool = True,
) -> PathwayBuild:
    """Fetch and assemble one pathway's internal graph.

    Oversized pathways are rejected rather than truncated: a randomly cut pathway
    is not a biologically meaningful object, and Reactome's hierarchy already
    provides smaller, complete subpathways to use instead.

    `prefilter` decides that from two cheap calls before any reaction is fetched.
    Without it, a pathway like Hemostasis costs several thousand requests — one
    per reaction, entity and nested member — only to be rejected at the end.
    """
    report: dict[str, Any] = {"st_id": st_id}

    if prefilter:
        size = R.pathway_size(st_id)
        report["prefilter_nodes"] = size.n_nodes
        if size.n_nodes > max_nodes:
            report["status"] = "too_large"
            return PathwayBuild(report=report)
        if size.n_nodes < min_nodes:
            report["status"] = "too_small"
            return PathwayBuild(report=report)

    event_ids = R.contained_event_ids(st_id)
    if not event_ids:
        report["status"] = "no_events"
        return PathwayBuild(report=report)

    reactions: dict[str, dict] = {}
    for rid in event_ids:
        obj = R.query(rid)
        if obj and obj.get("schemaClass") in EVENT_CLASSES:
            reactions[rid] = obj
    report["n_events_listed"] = len(event_ids)
    report["n_reactions_resolved"] = len(reactions)
    if not reactions:
        report["status"] = "no_reactions"
        return PathwayBuild(report=report)

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    entity_cache: dict[str, dict] = {}

    def add_entity(obj: dict) -> str | None:
        """Register an entity node, return its id. Deduplicated by stId."""
        ent_st = obj.get("stId")
        if not ent_st:
            return None
        nid = pathway_entity_id(st_id, str(ent_st))
        if nid in nodes:
            return nid
        schema_class = str(obj.get("schemaClass") or "PhysicalEntity")
        full = entity_cache.get(str(ent_st))
        if full is None:
            full = R.query(str(ent_st)) or obj
            entity_cache[str(ent_st)] = full
        nodes[nid] = Node(
            id=nid,
            type=entity_node_type(schema_class),
            name=str(obj.get("displayName") or full.get("displayName") or ent_st),
            attrs={
                "reactome_st_id": str(ent_st),
                "schema_class": schema_class,
                "compartments": _compartments(full),
                "identifiers": _ref_identifiers(full),
                "n_members": len(full.get("hasMember") or []) or None,
                "n_components": len(full.get("hasComponent") or []) or None,
            },
        )
        return nid

    for rid, obj in reactions.items():
        rnode = pathway_reaction_id(st_id, rid)
        nodes[rnode] = Node(
            id=rnode,
            type="reaction_event",
            name=str(obj.get("displayName") or rid),
            attrs={
                "reactome_st_id": rid,
                "schema_class": str(obj.get("schemaClass")),
                "compartments": _compartments(obj),
                "is_in_disease": bool(obj.get("isInDisease")),
                "is_inferred": bool(obj.get("isInferred")),
                "category": obj.get("category"),
            },
        )

    for rid, obj in reactions.items():
        rnode = pathway_reaction_id(st_id, rid)

        for item in obj.get("input") or []:
            if isinstance(item, dict) and (nid := add_entity(item)):
                edges.append(Edge(source=nid, target=rnode, type="input_to",
                                  attrs={"role": "input"}))
        for item in obj.get("output") or []:
            if isinstance(item, dict) and (nid := add_entity(item)):
                edges.append(Edge(source=rnode, target=nid, type="produces",
                                  attrs={"role": "output"}))

        if resolve_catalysts:
            for ca in obj.get("catalystActivity") or []:
                if not isinstance(ca, dict):
                    continue
                pe = ca.get("physicalEntity")
                if not isinstance(pe, dict) and ca.get("dbId"):
                    # CatalystActivity carries a dbId only; resolve it.
                    full = R.query(str(ca["dbId"])) or {}
                    pe = full.get("physicalEntity")
                if isinstance(pe, dict) and (nid := add_entity(pe)):
                    edges.append(Edge(source=rnode, target=nid, type="catalyzed_by",
                                      attrs={"activity": ca.get("displayName"),
                                             "role": "catalyst"}))

        for reg in obj.get("regulatedBy") or []:
            if not isinstance(reg, dict):
                continue
            cls = str(reg.get("schemaClass") or "")
            regulator = reg.get("regulator")
            if not isinstance(regulator, dict) and reg.get("dbId"):
                full = R.query(str(reg["dbId"])) or {}
                regulator = full.get("regulator")
                cls = cls or str(full.get("schemaClass") or "")
            if not isinstance(regulator, dict):
                continue
            etype = ("inhibited_by" if "Negative" in cls
                     else "activated_by" if "Positive" in cls else None)
            if etype and (nid := add_entity(regulator)):
                edges.append(Edge(source=rnode, target=nid, type=etype,
                                  attrs={"regulation_class": cls, "role": "regulator"}))

        # Curated ordering only. No inferred ordering: a `precedes` edge that
        # Reactome does not assert would be an invented biological claim.
        for prev in obj.get("precedingEvent") or []:
            if not isinstance(prev, dict):
                continue
            prev_st = str(prev.get("stId") or "")
            if prev_st in reactions:
                edges.append(Edge(
                    source=pathway_reaction_id(st_id, prev_st),
                    target=rnode, type="precedes",
                    attrs={"source": "reactome_precedingEvent"},
                ))

    # Deduplicate: the same participant can be listed twice on one reaction.
    seen: set[tuple[str, str, str]] = set()
    unique_edges: list[Edge] = []
    for e in edges:
        key = (e.source, e.type, e.target)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(e)

    report["n_nodes"] = len(nodes)
    report["n_edges"] = len(unique_edges)
    report["edge_type_counts"] = {
        t: sum(1 for e in unique_edges if e.type == t)
        for t in ("input_to", "produces", "catalyzed_by", "activated_by",
                  "inhibited_by", "precedes")
    }
    report["node_type_counts"] = {
        t: sum(1 for n in nodes.values() if n.type == t)
        for t in ("reaction_event", "molecular_state", "complex_state", "set_state")
    }

    if len(nodes) > max_nodes:
        report["status"] = "too_large"
        return PathwayBuild(report=report)
    if len(nodes) < min_nodes:
        report["status"] = "too_small"
        return PathwayBuild(report=report)

    isolated = [n for n in nodes if not any(n in (e.source, e.target) for e in unique_edges)]
    report["isolated_nodes"] = len(isolated)
    report["status"] = "ok"

    graph = InternalGraph(
        graph_id=pathway_graph_id(st_id),
        owner_root_id=root_id("pathway", st_id),
        kind="pathway",
        nodes=list(nodes.values()),
        edges=unique_edges,
        provenance={
            "st_id": st_id,
            "n_events_listed": len(event_ids),
            "n_reactions": len(reactions),
            "sets_kept_unexpanded": True,
            "ordering_source": "reactome_precedingEvent_only",
            "builder": "reactome_content_service/v1",
        },
    )
    return PathwayBuild(graph=graph, report=report)
