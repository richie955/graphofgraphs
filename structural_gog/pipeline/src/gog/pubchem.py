"""DrugBank id -> PubChem CID -> SMILES, plus the ChEBI xref used by Phase 4.

Why PubChem and not DrugBank's own structure dump: the dump is licence-restricted.
PubChem's xref index carries the DrugBank registry id, so the mapping is public
and citable. UniChem is kept as a fallback because PubChem's xref coverage for
DrugBank is good but not complete.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .cache import CachedSession
from .config import cfg

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
UNICHEM = "https://www.ebi.ac.uk/unichem/rest"

_session: CachedSession | None = None


def session() -> CachedSession:
    global _session
    if _session is None:
        _session = CachedSession("pubchem", cfg()["http"]["pubchem_min_interval"])
    return _session


@dataclass
class DrugStructure:
    drugbank_id: str
    cid: int | None = None
    smiles: str | None = None
    inchikey: str | None = None
    heavy_atoms: int | None = None       # PubChem's count; RDKit recount happens in Phase 2
    chebi_ids: list[str] | None = None
    resolved_via: str | None = None      # pubchem_xref | unichem
    note: str | None = None

    def ok(self) -> bool:
        return bool(self.smiles)

    def to_dict(self) -> dict:
        return asdict(self)


def cid_from_drugbank(db_id: str) -> tuple[int | None, str | None]:
    """Returns (cid, how). Lowest CID wins when the xref is ambiguous."""
    s = session()
    data = s.get_json(f"{PUBCHEM}/compound/xref/RegistryID/{db_id}/cids/JSON")
    cids = (data or {}).get("IdentifierList", {}).get("CID") or []
    if cids:
        return min(int(c) for c in cids), "pubchem_xref"

    # Fallback: UniChem source 2 is DrugBank, source 22 is PubChem.
    data = s.get_json(f"{UNICHEM}/src_compound_id/{db_id}/2/22")
    if isinstance(data, list) and data:
        try:
            return min(int(d["src_compound_id"]) for d in data), "unichem"
        except (KeyError, ValueError):
            pass
    return None, None


def properties(cid: int) -> dict:
    # PubChem renamed these in 2025: IsomericSMILES -> SMILES (stereo kept),
    # CanonicalSMILES -> ConnectivitySMILES. Legacy names are read as a fallback
    # in case the service is rolled back or mirrored.
    props = "SMILES,ConnectivitySMILES,InChIKey,HeavyAtomCount"
    data = session().get_json(f"{PUBCHEM}/compound/cid/{cid}/property/{props}/JSON")
    table = (data or {}).get("PropertyTable", {}).get("Properties") or [{}]
    return table[0]


def _pick_smiles(p: dict) -> str | None:
    for key in ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"):
        if p.get(key):
            return p[key]
    return None


def chebi_ids(cid: int) -> list[str]:
    """ChEBI accessions for a CID, used to match Reactome reaction participants."""
    data = session().get_json(
        f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON"
    )
    syns = (((data or {}).get("InformationList") or {}).get("Information") or [{}])[0]
    out = []
    for s in syns.get("Synonym", []) or []:
        if s.upper().startswith("CHEBI:"):
            out.append(s.upper())
    return sorted(set(out))


def resolve(db_id: str, *, want_chebi: bool = True) -> DrugStructure:
    rec = DrugStructure(drugbank_id=db_id)
    cid, how = cid_from_drugbank(db_id)
    if cid is None:
        rec.note = "no PubChem CID for this DrugBank id (often a biologic or a mixture)"
        return rec

    rec.cid, rec.resolved_via = cid, how
    p = properties(cid)
    rec.smiles = _pick_smiles(p)
    rec.inchikey = p.get("InChIKey")
    rec.heavy_atoms = p.get("HeavyAtomCount")
    if not rec.smiles:
        rec.note = "CID found but no SMILES property"
    if want_chebi:
        rec.chebi_ids = chebi_ids(cid)
    return rec
