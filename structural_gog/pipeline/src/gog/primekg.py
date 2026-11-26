"""PrimeKG loading, owner-node extraction, and the subset rule.

kg.csv columns: relation, display_relation,
                x_index, x_id, x_type, x_name, x_source,
                y_index, y_id, y_type, y_name, y_source
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from .config import cfg
from .schema import root_id

USECOLS = [
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
]


def load_primekg(path=None) -> pd.DataFrame:
    path = path or cfg()["paths"]["primekg_csv"]
    df = pd.read_csv(path, usecols=USECOLS, dtype=str, low_memory=False)
    for side in ("x", "y"):
        df[f"{side}_id"] = df[f"{side}_id"].str.strip()
        # PrimeKG writes some ids as floats ("5562.0"); normalise to the integer form.
        float_like = df[f"{side}_id"].str.fullmatch(r"\d+\.0")
        df.loc[float_like, f"{side}_id"] = df.loc[float_like, f"{side}_id"].str[:-2]
        df[f"{side}_node"] = df[f"{side}_type"] + ":" + df[f"{side}_id"]
    return df


@dataclass(frozen=True)
class OwnerNode:
    node_type: str      # drug | pathway
    source_id: str      # DB00945 | R-HSA-109582
    name: str
    source: str         # DrugBank | REACTOME

    @property
    def root_id(self) -> str:
        return root_id(self.node_type, self.source_id)


def node_table(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated node list from both edge endpoints."""
    frames = []
    for side in ("x", "y"):
        frames.append(
            df[[f"{side}_id", f"{side}_type", f"{side}_name", f"{side}_source"]].rename(
                columns={
                    f"{side}_id": "id", f"{side}_type": "type",
                    f"{side}_name": "name", f"{side}_source": "source",
                }
            )
        )
    nodes = pd.concat(frames, ignore_index=True)
    nodes["node"] = nodes["type"] + ":" + nodes["id"]
    return nodes.drop_duplicates(subset="node").reset_index(drop=True)


def owner_candidates(df: pd.DataFrame, kind: str) -> list[OwnerNode]:
    """Drug or pathway nodes whose source we trust (config: sources.*)."""
    conf = cfg()
    allowed = set(conf["sources"][kind])
    nodes = node_table(df)
    sel = nodes[(nodes["type"] == kind) & (nodes["source"].isin(allowed))]

    if kind == "pathway":
        prefix = conf["subset"]["pathway_species_prefix"]
        sel = sel[sel["id"].str.startswith(prefix)]

    return [
        OwnerNode(node_type=kind, source_id=r.id, name=str(r.name), source=str(r.source))
        for r in sel.itertuples()
    ]


def drug_target_proteins(df: pd.DataFrame) -> dict[str, set[str]]:
    """DrugBank id -> set of gene symbols PrimeKG links it to.

    Covers drug_protein in both directions. Gene *symbols* are used here only so
    the coverage estimate needs no UniProt round trip; the real cross-context
    builder should map Entrez -> UniProt and match on accession.
    """
    out: dict[str, set[str]] = {}
    mask = df["relation"].str.contains("drug_protein", na=False)
    sub = df[mask]
    for x_type, x_id, x_name, y_type, y_id, y_name in zip(
        sub["x_type"], sub["x_id"], sub["x_name"],
        sub["y_type"], sub["y_id"], sub["y_name"],
    ):
        if x_type == "drug" and y_type == "gene/protein":
            out.setdefault(x_id, set()).add(str(y_name).upper())
        elif y_type == "drug" and x_type == "gene/protein":
            out.setdefault(y_id, set()).add(str(x_name).upper())
    return out


# --------------------------------------------------------------------------- #
# Subset rule (plan §2). Seeds are owner nodes that survived the coverage
# filters; the subset is their k-hop neighbourhood, largest component only.
# --------------------------------------------------------------------------- #


def expand_subset(
    df: pd.DataFrame,
    seeds: set[str],
    hops: int,
    *,
    exclude_relations: set[str] | None = None,
) -> pd.DataFrame:
    """k-hop edge subset around `seeds`, done in pandas.

    Building a NetworkX graph of all 8.1M PrimeKG edges first needs roughly 10 GB;
    filtering the frame costs a few hundred MB. `exclude_relations` keeps
    hub relations such as drug_drug from dragging in the whole graph on hop 1
    (their edges are still kept at the end if both endpoints survive).
    """
    exclude_relations = exclude_relations or set()
    walkable = df[~df["relation"].isin(exclude_relations)] if exclude_relations else df

    frontier = set(seeds)
    keep = set(seeds)
    for _ in range(hops):
        touching = walkable["x_node"].isin(frontier) | walkable["y_node"].isin(frontier)
        hit = walkable[touching]
        nxt = set(hit["x_node"]) | set(hit["y_node"])
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if not frontier:
            break

    both_in = df["x_node"].isin(keep) & df["y_node"].isin(keep)
    return df[both_in]


def largest_component(edges: pd.DataFrame) -> pd.DataFrame:
    """Restrict an edge frame to its largest connected component."""
    g = nx.Graph()
    g.add_edges_from(zip(edges["x_node"], edges["y_node"]))
    if g.number_of_nodes() == 0:
        return edges
    comp = max(nx.connected_components(g), key=len)
    return edges[edges["x_node"].isin(comp) & edges["y_node"].isin(comp)]


def build_graph(df: pd.DataFrame) -> nx.MultiGraph:
    g = nx.MultiGraph()
    nodes = node_table(df)
    for r in nodes.itertuples():
        g.add_node(r.node, type=r.type, source_id=r.id, name=r.name, source=r.source)
    for r in df.itertuples():
        g.add_edge(r.x_node, r.y_node, relation=r.relation, display_relation=r.display_relation)
    return g


def subset_nodes(g: nx.MultiGraph, seeds: list[str], hops: int, largest_component: bool) -> set[str]:
    present = [s for s in seeds if s in g]
    keep: set[str] = set(present)
    frontier = set(present)
    for _ in range(hops):
        nxt: set[str] = set()
        for n in frontier:
            nxt |= set(g.neighbors(n))
        nxt -= keep
        keep |= nxt
        frontier = nxt

    if largest_component and keep:
        sub = g.subgraph(keep)
        keep = max(nx.connected_components(sub), key=len)
    return keep
