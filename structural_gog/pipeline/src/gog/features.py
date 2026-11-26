"""Input features for every node family.

Kept apart from the exporter so the vocabularies are one definition, not one per
framework: the same `atom_features()` has to produce identical vectors whether it
is called from the PyG export, a DGL export, or a notebook.

Fragment nodes get a one-hot over the fragment vocabulary rather than a nested
GNN over their atoms. It is simpler, it is interpretable — you can read which
group mattered — and it is the right first baseline: if a learned fragment
encoder cannot beat a one-hot label, the encoder is not earning its complexity.
"""
from __future__ import annotations

from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Atom features (~34 dims)
# --------------------------------------------------------------------------- #

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "OTHER"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "OTHER"]
NUM_HS = [0, 1, 2, 3, 4]

ATOM_FEATURE_NAMES: list[str] = (
    [f"element={e}" for e in ELEMENTS]
    + [f"degree={d}" for d in DEGREES]
    + [f"charge={c}" for c in CHARGES]
    + [f"hyb={h}" for h in HYBRIDIZATIONS]
    + [f"num_h={n}" for n in NUM_HS]
    + ["aromatic", "in_ring"]
)
ATOM_FEATURE_DIM = len(ATOM_FEATURE_NAMES)


def _one_hot(value: Any, choices: list[Any], *, other: bool = True) -> list[float]:
    vec = [0.0] * len(choices)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        if other and choices and choices[-1] in ("OTHER", "other"):
            vec[-1] = 1.0
    return vec


def atom_features(attrs: dict[str, Any]) -> list[float]:
    return (
        _one_hot(attrs.get("element"), ELEMENTS)
        + _one_hot(int(attrs.get("degree") or 0), DEGREES, other=False)
        + _one_hot(int(attrs.get("formal_charge") or 0), CHARGES, other=False)
        + _one_hot(str(attrs.get("hybridization") or "UNSPECIFIED").upper(), HYBRIDIZATIONS)
        + _one_hot(int(attrs.get("num_hs") or 0), NUM_HS, other=False)
        + [1.0 if attrs.get("aromatic") else 0.0, 1.0 if attrs.get("in_ring") else 0.0]
    )


# --------------------------------------------------------------------------- #
# Fragment features
# --------------------------------------------------------------------------- #

FRAGMENT_NUMERIC = ["n_atoms", "charge", "n_attachments", "is_ring_system"]


def fragment_vocabulary(graphs: Iterable[dict]) -> list[str]:
    """Labels seen in the data, sorted, plus OTHER. Built once and stored."""
    labels = set()
    for g in graphs:
        for n in g["nodes"]:
            if n["type"] == "molecular_fragment":
                labels.add(str((n.get("attrs") or {}).get("group_name") or "unknown"))
    return sorted(labels) + ["OTHER"]


def fragment_features(attrs: dict[str, Any], vocab: list[str]) -> list[float]:
    label = str(attrs.get("group_name") or "unknown")
    vec = [0.0] * len(vocab)
    vec[vocab.index(label) if label in vocab else len(vocab) - 1] = 1.0
    return vec + [
        float(attrs.get("n_atoms") or len(attrs.get("atoms") or [])),
        float(attrs.get("charge") or 0),
        float(len(attrs.get("attachment_atoms") or [])),
        1.0 if attrs.get("kind") == "ring_system" else 0.0,
    ]


# --------------------------------------------------------------------------- #
# Pathway entity / reaction features
# --------------------------------------------------------------------------- #

SCHEMA_CLASSES = [
    "SimpleEntity", "EntityWithAccessionedSequence", "Complex", "DefinedSet",
    "CandidateSet", "OpenSet", "GenomeEncodedEntity", "OtherEntity",
    "ChemicalDrug", "ProteinDrug", "RNADrug", "Polymer", "OTHER",
]
COMPARTMENTS = [
    "cytosol", "nucleoplasm", "extracellular region", "plasma membrane",
    "endoplasmic reticulum lumen", "endoplasmic reticulum membrane",
    "mitochondrial matrix", "mitochondrial intermembrane space",
    "Golgi lumen", "lysosomal lumen", "endosome lumen", "peroxisomal matrix",
    "nuclear envelope", "mitochondrial outer membrane", "OTHER",
]

PATHWAY_ENTITY_FEATURE_NAMES: list[str] = (
    [f"class={c}" for c in SCHEMA_CLASSES]
    + [f"compartment={c}" for c in COMPARTMENTS]
    + ["n_members", "n_components", "has_chebi", "has_uniprot", "has_gtopdb"]
)
PATHWAY_ENTITY_FEATURE_DIM = len(PATHWAY_ENTITY_FEATURE_NAMES)


def pathway_entity_features(attrs: dict[str, Any]) -> list[float]:
    compartments = attrs.get("compartments") or []
    comp_vec = [0.0] * len(COMPARTMENTS)
    for c in compartments:
        # Multi-hot: compartment is part of identity, and an entity can span two.
        idx = COMPARTMENTS.index(c) if c in COMPARTMENTS else len(COMPARTMENTS) - 1
        comp_vec[idx] = 1.0
    identifiers = attrs.get("identifiers") or {}
    return (
        _one_hot(str(attrs.get("schema_class") or "OTHER"), SCHEMA_CLASSES)
        + comp_vec
        + [
            float(attrs.get("n_members") or 0),
            float(attrs.get("n_components") or 0),
            1.0 if identifiers.get("chebi") else 0.0,
            1.0 if identifiers.get("uniprot") else 0.0,
            1.0 if identifiers.get("iuphar.ligand") else 0.0,
        ]
    )


REACTION_FEATURE_NAMES = (
    [f"class={c}" for c in ["Reaction", "BlackBoxEvent", "Polymerisation",
                            "Depolymerisation", "FailedReaction", "OTHER"]]
    + [f"compartment={c}" for c in COMPARTMENTS]
    + ["is_in_disease", "is_inferred"]
)
REACTION_FEATURE_DIM = len(REACTION_FEATURE_NAMES)


def reaction_features(attrs: dict[str, Any]) -> list[float]:
    classes = ["Reaction", "BlackBoxEvent", "Polymerisation",
               "Depolymerisation", "FailedReaction", "OTHER"]
    compartments = attrs.get("compartments") or []
    comp_vec = [0.0] * len(COMPARTMENTS)
    for c in compartments:
        idx = COMPARTMENTS.index(c) if c in COMPARTMENTS else len(COMPARTMENTS) - 1
        comp_vec[idx] = 1.0
    return (
        _one_hot(str(attrs.get("schema_class") or "OTHER"), classes)
        + comp_vec
        + [1.0 if attrs.get("is_in_disease") else 0.0,
           1.0 if attrs.get("is_inferred") else 0.0]
    )


BOND_TYPE_VOCAB = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def bond_features(attrs: dict[str, Any]) -> list[float]:
    return _one_hot(str(attrs.get("bond_type") or "SINGLE"), BOND_TYPE_VOCAB,
                    other=False) + [1.0 if attrs.get("conjugated") else 0.0]
