"""Phase 2: SMILES -> fragment-level molecular graph.

Decomposition, in order:

1. **Ring systems first.** Fused and spiro rings are merged into one system (any two
   rings sharing an atom belong together) and collapsed into a single fragment node.
   Doing rings first is what makes the rest unambiguous: no functional group can
   straddle a ring boundary afterwards.
2. **Functional groups second**, by SMARTS, most specific first. A group is accepted
   only when *every* one of its atoms is still unassigned, which is how overlaps
   resolve without arbitration — `C(=O)O` in benzoic acid is accepted because the
   ring carbons were already taken, and `[OX2H]` cannot then re-claim that oxygen.
3. **Everything left stays an atom node.**

A one-atom group (a phenol hydroxyl, a halide) becomes an attribute on the atom
rather than a fragment node: collapsing a single atom buys nothing and doubles the
node count.

The result is a partition — every heavy atom belongs to exactly one node — which is
what `validate_decomposition()` checks, along with rebuilding the original bond set
from the fragments and comparing it to RDKit's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from .schema import (
    BOND_TYPES, Edge, InternalGraph, Node,
    drug_atom_id, drug_fragment_id, drug_graph_id, drug_mol_ref_id, root_id,
)

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Functional groups, most specific first. Order IS the overlap policy.
# --------------------------------------------------------------------------- #

FUNCTIONAL_GROUPS: list[tuple[str, str]] = [
    ("carbamate",         "[NX3][CX3](=O)[OX2]"),
    ("urea",              "[NX3][CX3](=O)[NX3]"),
    ("guanidine",         "[NX3][CX3](=[NX2])[NX3]"),
    ("sulfonamide",       "[SX4](=O)(=O)[NX3]"),
    ("sulfonic_acid",     "[SX4](=O)(=O)[OX2H1]"),
    ("sulfone",           "[SX4](=O)(=O)"),
    ("sulfoxide",         "[SX3](=O)"),
    ("phosphate",         "[PX4](=O)([OX2])([OX2])"),
    ("phosphonate",       "[PX4](=O)([OX2])"),
    ("nitro",             "[NX3](=O)=O"),
    ("nitro_charged",     "[N+](=O)[O-]"),
    ("carboxylic_acid",   "[CX3](=O)[OX2H1]"),
    ("carboxylate",       "[CX3](=O)[OX1-]"),
    ("ester",             "[CX3](=O)[OX2][#6]"),
    ("amide",             "[CX3](=O)[NX3]"),
    ("anhydride",         "[CX3](=O)[OX2][CX3](=O)"),
    ("nitrile",           "[NX1]#[CX2]"),
    ("aldehyde",          "[CX3H1](=O)"),
    ("ketone",            "[#6][CX3](=O)[#6]"),
    ("imine",             "[CX3]=[NX2]"),
    ("azo",               "[NX2]=[NX2]"),
    ("ether",             "[OD2]([#6])[#6]"),
    ("thioether",         "[SD2]([#6])[#6]"),
    ("primary_amine",     "[NX3;H2;!$(N[CX3]=[OX1,SX1])]"),
    ("secondary_amine",   "[NX3;H1;!$(N[CX3]=[OX1,SX1]);!$(N=*)]"),
    ("tertiary_amine",    "[NX3;H0;!$(N[CX3]=[OX1,SX1]);!$(N=*);!$([N+])]"),
    ("hydroxyl",          "[OX2H]"),
    ("thiol",             "[SX2H]"),
    ("halide",            "[F,Cl,Br,I]"),
]

_COMPILED: list[tuple[str, Chem.Mol]] | None = None


def compiled_groups() -> list[tuple[str, Chem.Mol]]:
    global _COMPILED
    if _COMPILED is None:
        out = []
        for name, smarts in FUNCTIONAL_GROUPS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"bad SMARTS for {name}: {smarts}")
            out.append((name, patt))
        _COMPILED = out
    return _COMPILED


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #


@dataclass
class PreparedMol:
    mol: Chem.Mol
    canonical_smiles: str
    heavy_atoms: int
    stripped_salt: bool = False
    note: str | None = None


def prepare_mol(smiles: str) -> PreparedMol | None:
    """Parse, keep the largest covalent fragment, canonicalise.

    Salts and counter-ions are separate fragments in the SMILES; keeping them would
    put disconnected junk in the graph, so the largest piece by heavy-atom count wins.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    pieces = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    stripped = len(pieces) > 1
    if stripped:
        mol = max(pieces, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, Chem.KekulizeException):
        return None

    # Canonicalise, then re-parse: atom indices must match the SMILES we store,
    # or every atom id in the exported graph points at the wrong atom.
    canonical = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    return PreparedMol(
        mol=mol,
        canonical_smiles=canonical,
        heavy_atoms=mol.GetNumHeavyAtoms(),
        stripped_salt=stripped,
    )


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #


@dataclass
class Fragment:
    index: int
    label: str
    kind: str                      # ring_system | functional_group
    atoms: list[int]
    internal_bonds: list[tuple[int, int, str]] = field(default_factory=list)
    charge: int = 0
    attachment_atoms: list[int] = field(default_factory=list)


@dataclass
class Decomposition:
    prepared: PreparedMol
    fragments: list[Fragment]
    atom_nodes: list[int]                        # heavy atoms kept as their own node
    atom_labels: dict[int, str]                  # one-atom group annotations
    owner: dict[int, str]                        # atom index -> node id
    problems: list[str] = field(default_factory=list)


def ring_systems(mol: Chem.Mol) -> list[list[int]]:
    """Fused and spiro rings merged: any two rings sharing an atom are one system."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    merged: list[set[int]] = []
    for r in rings:
        hits = [m for m in merged if m & r]
        if not hits:
            merged.append(set(r))
            continue
        combined = set(r)
        for m in hits:
            combined |= m
            merged.remove(m)
        merged.append(combined)
    return [sorted(m) for m in merged]


def decompose(prepared: PreparedMol) -> Decomposition:
    mol = prepared.mol
    assigned: dict[int, int] = {}          # atom index -> fragment index
    fragments: list[Fragment] = []
    atom_labels: dict[int, str] = {}

    def claim(atoms: list[int], label: str, kind: str) -> None:
        idx = len(fragments)
        fragments.append(Fragment(index=idx, label=label, kind=kind, atoms=sorted(atoms)))
        for a in atoms:
            assigned[a] = idx

    for system in ring_systems(mol):
        aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in system)
        n_rings = sum(1 for r in mol.GetRingInfo().AtomRings() if set(r) <= set(system))
        label = ("aromatic_ring" if aromatic else "ring") + (f"_system_{n_rings}" if n_rings > 1 else "")
        claim(system, label, "ring_system")

    for name, patt in compiled_groups():
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            atoms = list(match)
            if any(a in assigned for a in atoms):
                continue               # overlap: an earlier, more specific claim wins
            if len(atoms) == 1:
                atom_labels[atoms[0]] = name      # annotate, do not collapse
                continue
            claim(atoms, name, "functional_group")

    atom_nodes = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in assigned]

    # Fragment bookkeeping: internal bonds, charge, and which atoms reach outside.
    owner: dict[int, str] = {}
    for a in atom_nodes:
        owner[a] = str(a)
    for f in fragments:
        for a in f.atoms:
            owner[a] = f"F{f.index}"
    for f in fragments:
        members = set(f.atoms)
        f.charge = sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in f.atoms)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in members and j in members:
                f.internal_bonds.append((i, j, str(b.GetBondType())))
            elif i in members or j in members:
                inside = i if i in members else j
                if inside not in f.attachment_atoms:
                    f.attachment_atoms.append(inside)
        f.attachment_atoms.sort()

    d = Decomposition(prepared=prepared, fragments=fragments, atom_nodes=atom_nodes,
                      atom_labels=atom_labels, owner=owner)
    d.problems = validate_decomposition(d)
    return d


def validate_decomposition(d: Decomposition) -> list[str]:
    """Phase 6 checks for one molecule: partition completeness and bond recovery."""
    problems: list[str] = []
    mol = d.prepared.mol
    n = mol.GetNumAtoms()

    covered: list[int] = list(d.atom_nodes)
    for f in d.fragments:
        covered.extend(f.atoms)
    if len(covered) != n:
        problems.append(f"covered {len(covered)} atoms, molecule has {n}")
    if len(set(covered)) != len(covered):
        problems.append("an atom belongs to more than one node")
    if set(covered) != set(range(n)):
        missing = sorted(set(range(n)) - set(covered))
        problems.append(f"atoms never assigned: {missing[:10]}")

    # Rebuild every bond from fragment internals + cross-node bonds.
    rebuilt: set[tuple[int, int]] = set()
    for f in d.fragments:
        for i, j, _ in f.internal_bonds:
            rebuilt.add((min(i, j), max(i, j)))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if d.owner.get(i) != d.owner.get(j):
            rebuilt.add((min(i, j), max(i, j)))
    original = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                 max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    if rebuilt != original:
        problems.append(
            f"bond reconstruction mismatch: {len(original - rebuilt)} lost, "
            f"{len(rebuilt - original)} invented"
        )
    return problems


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #


def atom_attrs(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "element": atom.GetSymbol(),
        "atomic_num": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "formal_charge": atom.GetFormalCharge(),
        "aromatic": atom.GetIsAromatic(),
        "hybridization": str(atom.GetHybridization()),
        "in_ring": atom.IsInRing(),
        "num_hs": atom.GetTotalNumHs(),
    }


def build_drug_graph(db_id: str, smiles: str, *, pubchem_cid: int | None = None,
                     max_heavy_atoms: int = 150) -> tuple[InternalGraph | None, dict[str, Any]]:
    """Returns (graph, report). `graph` is None when the molecule is rejected."""
    report: dict[str, Any] = {"drugbank_id": db_id}

    prepared = prepare_mol(smiles)
    if prepared is None:
        report["status"] = "unparseable"
        return None, report
    report["heavy_atoms"] = prepared.heavy_atoms
    report["stripped_salt"] = prepared.stripped_salt
    if prepared.heavy_atoms > max_heavy_atoms:
        report["status"] = "too_large"
        return None, report
    if prepared.heavy_atoms < 2:
        report["status"] = "too_small"
        return None, report

    d = decompose(prepared)
    report["status"] = "ok" if not d.problems else "invalid"
    report["problems"] = d.problems
    report["n_fragments"] = len(d.fragments)
    report["n_atom_nodes"] = len(d.atom_nodes)
    report["fragment_labels"] = sorted({f.label for f in d.fragments})
    if d.problems:
        return None, report

    graph_id = drug_graph_id(db_id)
    mol = prepared.mol
    nodes: list[Node] = []
    node_of: dict[int, str] = {}

    for a in d.atom_nodes:
        atom = mol.GetAtomWithIdx(a)
        nid = drug_atom_id(db_id, a)
        node_of[a] = nid
        attrs = atom_attrs(atom)
        if a in d.atom_labels:
            attrs["functional_group"] = d.atom_labels[a]
        nodes.append(Node(id=nid, type="atom", name=atom.GetSymbol(), attrs=attrs))

    for f in d.fragments:
        nid = drug_fragment_id(db_id, f.index)
        for a in f.atoms:
            node_of[a] = nid
        nodes.append(Node(
            id=nid, type="molecular_fragment", name=f.label,
            attrs={
                "group_name": f.label,
                "kind": f.kind,
                "atoms": f.atoms,
                "atom_elements": [mol.GetAtomWithIdx(a).GetSymbol() for a in f.atoms],
                "internal_bonds": [[i, j, t] for i, j, t in f.internal_bonds],
                "charge": f.charge,
                "attachment_atoms": f.attachment_atoms,
                "n_atoms": len(f.atoms),
            },
        ))

    # Reference anchor for Phase 4 `same_chemical_as`. Deliberately edge-free:
    # it represents the whole molecule, so it is excluded from molecular pooling.
    nodes.append(Node(
        id=drug_mol_ref_id(db_id), type="molecule_ref", name=db_id,
        attrs={"canonical_smiles": prepared.canonical_smiles, "pubchem_cid": pubchem_cid,
               "exclude_from_pooling": True},
    ))

    edges: list[Edge] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src, dst = node_of[i], node_of[j]
        if src == dst:
            continue                       # internal to a fragment
        bond_type = str(b.GetBondType())
        both_atoms = (i in d.atom_nodes) and (j in d.atom_nodes)
        edges.append(Edge(
            source=src, target=dst,
            type="chemical_bond" if both_atoms else "chemical_attachment",
            attrs={
                "bond_type": bond_type if bond_type in BOND_TYPES else "SINGLE",
                "raw_bond_type": bond_type,
                "attach_from_atom": i,
                "attach_to_atom": j,
                "conjugated": b.GetIsConjugated(),
            },
        ))

    graph = InternalGraph(
        graph_id=graph_id,
        owner_root_id=root_id("drug", db_id),
        kind="drug",
        nodes=nodes,
        edges=edges,
        provenance={
            "canonical_smiles": prepared.canonical_smiles,
            "input_smiles": smiles,
            "pubchem_cid": pubchem_cid,
            "heavy_atoms": prepared.heavy_atoms,
            "stripped_salt": prepared.stripped_salt,
            "fragmenter": "ring_systems+smarts_functional_groups/v1",
            "n_functional_group_patterns": len(FUNCTIONAL_GROUPS),
            "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
        },
    )
    report["n_nodes"] = len(nodes)
    report["n_edges"] = len(edges)
    return graph, report
