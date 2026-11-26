"""Colab patch: add a gated residual from the Stage-1 state to the encoder output.

Paste the contents of this file into a Colab cell and run it after cloning the repo.

WHY
---
`GoGEncoder.forward` currently does:

    b_entity, _, z = self.initializer(...)      # sharp Stage-1 state
    for _ in range(self.num_rounds):
        ...
        b_entity, z, h_graph = b_new, z_next, h_fused   # z overwritten every round
    return GoGEncoderOutput(b_entity, z, h_graph, rounds)

The Stage-1 state never reaches the decoder. Measured consequence: on root-only data the
encoder scores filtered MRR 0.0927 against the entity-only baseline's 0.2051 — message
passing costs 54.8%, because propagation averages each entity's position toward its
neighbourhood while ComplEx keeps it sharp.

The baseline is therefore NOT a special case of this architecture, so the architecture
cannot be at least as good as it. A gated residual fixes that:

    gate = sigmoid(W [z0 ; z_final])
    z    = gate * z_final + (1 - gate) * z0

At gate -> 0 the encoder reproduces the baseline exactly. At gate -> 1 it is unchanged
from today. Anywhere between, it keeps ComplEx-like sharpness where propagation hurts and
takes structure where it helps — and the structural level is known to help: adding the
internal graphs improved filtered MRR 50.5%, Hits@10 59.6%.

The gate is initialised with a negative bias so training starts near the baseline and has
to earn its way toward propagation, rather than starting smoothed.
"""

import re
from pathlib import Path

import torch
from torch import nn

TARGET = Path("/content/gog/gog_embedding/__init__.py")
src = TARGET.read_text()

if "skip_gate" in src:
    print("already patched")
else:
    # 1. Register the gate on GoGEncoder. Its __init__ ends with the stage modules;
    #    anchor on the Stage 8 / feedback assignment which is unique to this class.
    anchor = re.search(r"(\n\s+self\.feedback\s*=\s*Stage8ContextFeedback\([^\n]*\)\n)", src)
    if anchor is None:
        raise SystemExit("could not find the Stage 8 assignment; inspect GoGEncoder.__init__")
    insert_at = anchor.end(1)
    src = src[:insert_at] + (
        "        self.skip_gate = nn.Linear(2 * hidden_dim, hidden_dim)\n"
        "        nn.init.zeros_(self.skip_gate.weight)\n"
        "        nn.init.constant_(self.skip_gate.bias, -2.0)\n"
    ) + src[insert_at:]

    # 2. Keep the Stage-1 occurrence state.
    src = src.replace(
        "        b_entity, _, z = self.initializer(entity_types, entity_by_occ)\n"
        "        h_graph = z.new_zeros",
        "        b_entity, _, z = self.initializer(entity_types, entity_by_occ)\n"
        "        z_initial = z\n"
        "        h_graph = z.new_zeros",
        1,
    )

    # 3. Mix it back in before returning.
    src = src.replace(
        "            b_entity, z, h_graph = b_new, z_next, h_fused\n"
        "        return GoGEncoderOutput(b_entity, z, h_graph, rounds)",
        "            b_entity, z, h_graph = b_new, z_next, h_fused\n"
        "        gate = torch.sigmoid(self.skip_gate(torch.cat((z_initial, z), dim=-1)))\n"
        "        z = gate * z + (1.0 - gate) * z_initial\n"
        "        return GoGEncoderOutput(b_entity, z, h_graph, rounds)",
        1,
    )
    TARGET.write_text(src)
    print("patched gog_embedding/__init__.py")

# Verify all three edits landed.
check = TARGET.read_text()
for needle, label in (("self.skip_gate = nn.Linear", "gate module"),
                      ("z_initial = z", "Stage-1 state kept"),
                      ("gate * z + (1.0 - gate) * z_initial", "residual mix")):
    print(f"  {'OK  ' if needle in check else 'MISSING'}  {label}")
