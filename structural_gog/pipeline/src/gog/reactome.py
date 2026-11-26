"""Reactome Content Service queries needed for sizing and for Phases 3 and 4.

Endpoints used:
  /data/pathway/{stId}/containedEvents   nested events as objects (stId, schemaClass, displayName)
  /data/participants/{stId}              participants, each with its reference entities
  /data/query/{stId}                     full object (class, compartment, regulation)

Response shapes verified live, 2026-11. Two traps worth knowing:

* `containedEvents/stId` returns text/plain, not JSON, and is rejected under an
  `Accept: application/json` header. Use the plain `/containedEvents` variant.
* A participant carries `peDbId` (singular). Its `refEntities` have no
  `databaseName` field; the database is the prefix of `stId`, e.g.
  `uniprot:Q8N8N7`, `chebi:57400`, `iuphar.ligand:4194`.

The third prefix is the one that matters for cross-context edges: Reactome
curates **drugs as `ReferenceTherapeutic` with Guide to Pharmacology ids**, not
as ChEBI molecules. Matching a drug on ChEBI alone will miss most of the
curated drug participants.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cache import CachedSession
from .config import cfg

BASE = "https://reactome.org/ContentService"

_session: CachedSession | None = None


def session() -> CachedSession:
    global _session
    if _session is None:
        _session = CachedSession("reactome", cfg()["http"]["reactome_min_interval"])
    return _session


# --------------------------------------------------------------------------- #
# Raw queries
# --------------------------------------------------------------------------- #


def contained_events(st_id: str) -> list[dict]:
    """Every event nested under a pathway, at any depth, as objects."""
    data = session().get_json(f"{BASE}/data/pathway/{st_id}/containedEvents")
    return data if isinstance(data, list) else []


def contained_event_ids(st_id: str) -> list[str]:
    """Reaction stIds nested under a pathway, at any depth.

    Uses the `/stId` variant, which serves text/plain. Its list is flat and
    complete, unlike `/containedEvents`, whose JSON replaces repeated objects
    with bare dbId integers.
    """
    text = session().get_text(f"{BASE}/data/pathway/{st_id}/containedEvents/stId")
    if not text:
        return []
    return [t.strip() for t in text.strip().strip("[]").split(",") if t.strip()]


def participants(st_id: str) -> list[dict]:
    """Participants of a pathway or reaction, each with its reference entities."""
    data = session().get_json(f"{BASE}/data/participants/{st_id}")
    return data if isinstance(data, list) else []


def query(st_id: str) -> dict:
    data = session().get_json(f"{BASE}/data/query/{st_id}")
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# Reference-entity parsing
# --------------------------------------------------------------------------- #

# "UniProt:Q8N8N7 PTGR2"                              -> gene symbol PTGR2
_RE_UNIPROT_NAME = re.compile(r"^UniProt:\S+\s+(\S+)")
# "flurbiprofen [Guide to Pharmacology:4194]"         -> flurbiprofen
# "15-dehydro-prostaglandin E2(1-) [ChEBI:57400]"     -> 15-dehydro-prostaglandin E2(1-)
_RE_BRACKET_NAME = re.compile(r"^(.*?)\s*\[[^\]]+\]\s*$")


def ref_database(ref: dict) -> str:
    """'uniprot' | 'chebi' | 'iuphar.ligand' | '' — taken from the stId prefix."""
    st = str(ref.get("stId") or "")
    return st.split(":", 1)[0].lower() if ":" in st else ""


def ref_label(ref: dict) -> str | None:
    """The human name, stripped of its bracketed database tag."""
    disp = str(ref.get("displayName") or "")
    m = _RE_BRACKET_NAME.match(disp)
    if m:
        return m.group(1).strip()
    m = _RE_UNIPROT_NAME.match(disp)
    if m:
        return m.group(1).strip()
    return disp or None


def normalise_drug_name(name: str) -> str:
    """Loose key for name-based drug matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


# --------------------------------------------------------------------------- #
# Sizing + identifier harvest
# --------------------------------------------------------------------------- #


@dataclass
class PathwaySize:
    st_id: str
    n_events: int = 0
    n_entities: int = 0
    chebi: set[str] = field(default_factory=set)          # CHEBI:57400
    uniprot: set[str] = field(default_factory=set)        # Q8N8N7
    gene_names: set[str] = field(default_factory=set)     # PTGR2
    therapeutics: dict[str, str] = field(default_factory=dict)  # gtp id -> drug name
    therapeutic_names: set[str] = field(default_factory=set)    # normalised names
    schema_classes: dict[str, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.n_events + self.n_entities


def pathway_size(st_id: str) -> PathwaySize:
    """Node-count estimate plus the identifier sets Phase 4 matches against.

    n_nodes is events + distinct physical entities, which is what the Phase 3
    internal graph holds before any DefinedSet/CandidateSet expansion. Treat it
    as a lower bound for pathways with many sets.
    """
    size = PathwaySize(st_id=st_id)

    events = contained_events(st_id)
    size.n_events = len(events)
    for e in events:
        # Only the first occurrence of an object is expanded; later ones are
        # bare dbId integers (see the module docstring).
        if not isinstance(e, dict):
            size.schema_classes["_ref"] = size.schema_classes.get("_ref", 0) + 1
            continue
        cls = str(e.get("schemaClass") or "Event")
        size.schema_classes[cls] = size.schema_classes.get(cls, 0) + 1

    entity_ids: set[str] = set()
    for p in participants(st_id):
        if not isinstance(p, dict):
            entity_ids.add(str(p))
            continue
        pe = p.get("peDbId")
        if pe is not None:
            entity_ids.add(str(pe))
        for ref in p.get("refEntities") or []:
            if not isinstance(ref, dict):
                continue
            db = ref_database(ref)
            ident = str(ref.get("identifier") or "")
            label = ref_label(ref)
            if not ident:
                continue
            if db == "chebi":
                size.chebi.add(f"CHEBI:{ident}")
            elif db == "uniprot":
                size.uniprot.add(ident)
                if label:
                    size.gene_names.add(label.upper())
            elif db == "iuphar.ligand":
                # Reactome's curated drug participants live here.
                size.therapeutics[ident] = label or ident
                if label:
                    size.therapeutic_names.add(normalise_drug_name(label))
    size.n_entities = len(entity_ids)
    return size
