"""Model-side structural categories and graph-family readout.

Only explicit atom symbols and builder fragment labels are used. No chemical
attributes are guessed from free-text reaction or molecular-state names.
"""

import re

import torch
from torch import Tensor, nn

from data_pipeline.preprocessing import PreprocessedGoG
from .stage4 import Stage4GraphReadout


ELEMENTS = ("UNK " + "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
            "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac "
            "Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
GROUPS = ("carbamate urea guanidine sulfonamide sulfonic_acid sulfone sulfoxide phosphate phosphonate "
          "nitro nitro_charged carboxylic_acid carboxylate ester amide anhydride nitrile aldehyde ketone "
          "imine azo ether thioether primary_amine secondary_amine tertiary_amine hydroxyl thiol halide").split()
FRAGMENTS = ["UNK", *GROUPS, "ring", "aromatic_ring", "ring_system", "aromatic_ring_system"]


class StructuralCategories(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.element = nn.Embedding(len(ELEMENTS), hidden_dim)
        self.fragment = nn.Embedding(len(FRAGMENTS), hidden_dim)
        # Ring multiplicity is an explicit part of the builder label, not a
        # property inferred from a biochemical name.
        self.ring_count = nn.Linear(1, hidden_dim, bias=False)
        nn.init.normal_(self.element.weight, std=0.02)
        nn.init.normal_(self.fragment.weight, std=0.02)

    @staticmethod
    def categories(entity: dict) -> tuple[bool, int, bool, int, float]:
        kind = str(entity.get("type", entity.get("type_id", "")))
        name = str(entity.get("name", ""))
        atom = kind == "atom" or kind.startswith("atom_")
        fragment = kind == "molecular_fragment" or kind.startswith("frag_")
        element_id = ELEMENTS.index(name) if atom and name in ELEMENTS else 0
        label, count = name, 0.0
        match = re.fullmatch(r"(aromatic_ring|ring)_system_([2-9]|[1-9][0-9]{1,2})", name)
        if match:
            label, count = match[1] + "_system", float(match[2])
        fragment_id = FRAGMENTS.index(label) if fragment and label in FRAGMENTS else 0
        return atom, element_id, fragment, fragment_id, count

    def forward(self, data: PreprocessedGoG) -> Tensor:
        device = self.element.weight.device
        rows = [self.categories(e) for e in data.entities]
        values = torch.tensor(rows, device=device, dtype=self.element.weight.dtype).reshape(-1, 5)
        return (self.element(values[:, 1].long()) * values[:, 0:1]
                + self.fragment(values[:, 3].long()) * values[:, 2:3]
                + self.ring_count(values[:, 4:5].log1p()) * values[:, 2:3])


class FamilyGraphReadout(nn.Module):
    """Separate molecular/pathway heads; generic head for legacy fixture owners."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Stage4GraphReadout(hidden_dim, dropout, count_sensitive=True)
                                    for _ in range(3)])

    def forward(self, z: Tensor, member_graph: Tensor, member_occ: Tensor,
                previous: Tensor, families: Tensor) -> Tensor:
        result = previous
        for family, head in enumerate(self.heads):
            members = families[member_graph] == family
            updated, _, _ = head(z, member_graph[members], member_occ[members], previous)
            result = torch.where((families == family)[:, None], updated, result)
        return result
