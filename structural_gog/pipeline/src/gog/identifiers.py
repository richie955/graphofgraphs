"""Chemical and protein identifier mapping for Phase 4.

UniChem (EBI) maps a DrugBank id onto every other registry that shares the
compound's standard InChIKey, which is what gives `participates_in_reaction`
real evidence instead of a name guess.

Coverage is partial and that is expected: UniChem links aspirin and flurbiprofen
to Guide to Pharmacology but not abacavir, even though Reactome curates abacavir
as a therapeutic. So Phase 4 uses three methods in descending strength —
Guide to Pharmacology id, ChEBI accession, then a name match restricted to
Reactome entities that are already typed as therapeutics — and records which one
produced each edge.

The v1 API is a POST; the old `/unichem/rest/...` paths now 404.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import requests

from .cache import JsonCache, RateLimiter
from .config import cfg

UNICHEM_V1 = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
SRC_DRUGBANK = 2

UNIPROT_IDMAP = "https://rest.uniprot.org/idmapping"

_cache: JsonCache | None = None
_limiter: RateLimiter | None = None
_session: requests.Session | None = None


def _setup():
    global _cache, _limiter, _session
    if _cache is None:
        _cache = JsonCache("unichem")
        _limiter = RateLimiter(0.2)
        _session = requests.Session()
        _session.headers["User-Agent"] = "gog-pipeline/0.1 (research; python-requests)"
    return _cache, _limiter, _session


@dataclass
class ChemXrefs:
    drugbank_id: str
    inchikey: str | None = None
    sources: dict[str, str] = field(default_factory=dict)   # shortName -> compoundId

    @property
    def gtopdb(self) -> str | None:
        return self.sources.get("gtopdb")

    @property
    def chebi(self) -> str | None:
        v = self.sources.get("chebi")
        return v.replace("CHEBI:", "") if v else None


def unichem_xrefs(db_id: str) -> ChemXrefs:
    cache, limiter, session = _setup()
    key = f"unichem:v1:drugbank:{db_id}"
    hit = cache.get(key)
    if hit is None and not cache._path(key).exists():
        limiter.wait()
        try:
            r = session.post(
                UNICHEM_V1,
                json={"type": "sourceID", "compound": db_id, "sourceID": SRC_DRUGBANK},
                timeout=cfg()["http"]["timeout"],
            )
            hit = r.json() if r.ok else {}
        except (requests.RequestException, ValueError):
            hit = {}
        cache.put(key, hit)
    hit = hit or {}

    comps = hit.get("compounds") or []
    out = ChemXrefs(drugbank_id=db_id)
    if not comps:
        return out
    c = comps[0]
    out.inchikey = c.get("standardInchiKey")
    for s in c.get("sources") or []:
        name, cid = s.get("shortName"), s.get("compoundId")
        if name and cid:
            out.sources[str(name)] = str(cid)
    return out


# --------------------------------------------------------------------------- #
# Entrez gene id -> UniProt accession
# --------------------------------------------------------------------------- #


def entrez_to_uniprot(gene_ids: list[str], *, batch: int = 10) -> dict[str, list[str]]:
    """Map Entrez gene ids to reviewed human UniProt accessions.

    Uses UniProt's search endpoint rather than the async id-mapping job: one
    query per batch, no polling, and the result is cached. Reviewed entries are
    preferred because a gene with twenty unreviewed isoforms would otherwise
    produce twenty spurious `same_entity_as` edges.

    Batches are small on purpose. A long `OR` chain is accepted with status 200
    but silently returns almost nothing — 400 ids came back with 2 hits, 10 ids
    come back with 10 — so an oversized batch looks like a coverage problem
    rather than a malformed query. If a batch returns far fewer hits than ids,
    it is retried one id at a time.
    """
    cache, limiter, session = _setup()
    out: dict[str, list[str]] = {}
    todo: list[str] = []
    for gid in gene_ids:
        hit = cache.get(f"uniprot:entrez:{gid}")
        if hit is not None or cache._path(f"uniprot:entrez:{gid}").exists():
            out[gid] = hit or []
        else:
            todo.append(gid)

    def fetch(chunk: list[str]) -> dict[str, list[str]]:
        query = " OR ".join(f"xref:geneid-{g}" for g in chunk)
        url = (
            "https://rest.uniprot.org/uniprotkb/search"
            f"?query=({query}) AND (organism_id:9606) AND (reviewed:true)"
            "&fields=accession,xref_geneid&format=json&size=500"
        )
        limiter.wait()
        try:
            r = session.get(url, timeout=cfg()["http"]["timeout"])
            data = r.json() if r.ok else {}
        except (requests.RequestException, ValueError):
            data = {}

        found: dict[str, list[str]] = {}
        wanted = set(chunk)
        for entry in data.get("results") or []:
            acc = entry.get("primaryAccession")
            if not acc:
                continue
            for xref in entry.get("uniProtKBCrossReferences") or []:
                if xref.get("database") != "GeneID":
                    continue
                gid = str(xref.get("id") or "").split(".")[0]
                if gid in wanted:
                    found.setdefault(gid, [])
                    if acc not in found[gid]:
                        found[gid].append(acc)
        return found

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        found = fetch(chunk)
        if len(found) * 2 < len(chunk) and len(chunk) > 1:
            # Suspiciously empty: fall back to one id per request for this chunk.
            found = {}
            for gid in chunk:
                found.update(fetch([gid]))
        for gid in chunk:
            accs = found.get(gid, [])
            cache.put(f"uniprot:entrez:{gid}", accs)
            out[gid] = accs
    return out
