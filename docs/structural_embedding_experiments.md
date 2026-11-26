# Structural embedding experiments

This update changes the encoder and trainer, not graph construction or the supplied
JSON files. Smoke results verify implementation behavior; they do not establish a
biomedical prediction improvement.

## Profiles and inputs

`legacy` retains the original two-round architecture and state-dictionary keys.
`structural` defaults to one pass through stages 2–6. It adds shared embeddings for
explicit exported atom symbols and validated fragment labels, family-specific
attention/sum/count readouts, and a ROOT identity residual. It does not reconstruct
missing chemical attributes from names, fetch data, or merge distinct Reactome states.

The structural message defaults are `--basis-init per_matrix`, `--composition residual`
(`sender * (1 + relation_vector)`), and `--normalization relation` (mean per
receiver/relation/direction, then sum). Every stored edge creates exactly one forward
and one reverse computational channel. Legacy message settings remain selectable.

For ROOT occurrences, the decoder receives
`z_initial + identity_scale[type] * (z_encoded - z_initial)`. Scales start at zero;
they are unconstrained residual scales, not probabilities. New graph parameters get
no gradient through this path on the first zero-scale step; after the scales move,
the graph path can learn. `--no-identity-residual` removes this path for an ablation.

`--feedback --rounds 2` enables residual feedback only between rounds; `--sync`
additionally synchronizes entities with multiple occurrences. Synchronization requires
feedback. The feedback residual starts at 0.1, and no final reconstruction is performed.
Downward owner fusion is disabled when feedback is disabled.

## Training and checkpoint transfer

Install training dependencies in a workspace virtual environment, keeping the existing
PyTorch/CUDA installation when appropriate:

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install -r requirements-training.txt
```

The structural profile defaults to `--task-scope root`: only supervised ROOT facts
are prediction targets and receive negatives. Existing internal/cross-context training
edges remain message-passing inputs, even when the supplied JSON lacks input-only
flags. Every model in the comparison uses this same objective. `--task-scope all`
retains the legacy target universe and respects `derived_channel`/`supervised` flags.

ROOT-mode candidates are **all ROOT occurrences**, including when an optional relation
schema exists. Filtering and optimistic tie handling are unchanged. Validation filters
train+validation known positives; test filtering includes train+validation+test.

One epoch remains **one optimizer update**; decoder batches accumulate gradients before
the encoder backward pass and optimizer step. Smaller batches change memory use, not
the update budget. `--embedding-learning-rate`, `--encoder-learning-rate`, and
`--decoder-learning-rate` default to `--learning-rate` when omitted.

`--init-checkpoint PATH` transfers a baseline's entity/type/relation rows by IDs and
copies occurrence normalization and decoder projection. Missing ROOT identities,
types, or relations are errors. New checkpoints include type mappings; historical
baseline files without them require identical entity mappings. Exact legacy GoG
restoration requires identical mappings. Optimizers reset on initialization, and the
starting validation checkpoint remains eligible for selection at phase epoch zero.

## Bounded smoke comparison

From the repository root:

```powershell
.venv/Scripts/python.exe -m training.experiments docs/synthetic_gog_fixture.json --device cpu --threads 2 --hidden-dim 16 --num-bases 2 --seeds 17 --warmup-updates 2 --continuation-updates 2 --negatives 2 --batch-size 4 --output-dir runs/structural-smoke
```

The runner trains a shared baseline for two updates, continues it for two, trains
R-GCN and legacy GoG independently for four each, and adapts structural GoG for two
after transfer. It checks identical query/candidate fingerprints and negative schedules.
Test metrics are absent unless `--evaluate-test` is explicitly passed.

## Full comparison commands — not run during implementation

```powershell
.venv/Scripts/python.exe -m training.experiments structural_gog/gog/gog_model_subset_coarse.json --device cuda --output-dir runs/structural-subset
```

Defaults: hidden size 128, eight bases, four heads, dropout 0.2, 50 negatives per side,
batch size 4096, learning rates 0.001, weight decay 0.0001, gradient clipping 1.0,
seeds 17/23/42. Shared baseline pretraining has 80 updates; continuation and structural
adaptation have another 80; R-GCN and legacy GoG each have 160. Budgets are fixed
(no early termination), but validation selects checkpoints. Shared pretraining cost
and each branch's actual runtime are reported separately; equal update caps do not
imply equal wall-clock compute. Run validation comparisons first, freeze the experiment
choices, then add `--evaluate-test` for the explicit final evaluation.

Targeted additional configurations can be requested with, for example:

```powershell
.venv/Scripts/python.exe -m training.experiments structural_gog/gog/gog_model_subset_coarse.json --device cuda --ablations identity_off chemical_off message_legacy feedback sync_feedback --output-dir runs/structural-ablations
```

`--ablations` also supports `basis_legacy`, `composition_multiply`, `normalization_global`,
`readout_shared`, `fusion_off`, `local_off`, `cross_off`, and `root_off`. The reference
models retain their original architectures; these settings affect the added structural
branches. This is a selected set of experiments, not an exhaustive combination sweep.

Individual runs remain available via `python -m training.train DATA --model gog
--encoder-profile structural`. Use `--fixed-budget` to disable early stopping. The
legacy command/API retain legacy defaults; structural runs are validation-only unless
`--evaluate-test` is supplied.

## Reports and checks

Each suite writes `manifest.json`, `comparison.csv`, `comparison.json`, per-model JSON
reports, and best checkpoints. Reports include configuration, ID mappings in checkpoints,
dataset/query/candidate fingerprints, negative-sample fingerprints, initialization
provenance, actual optimizer steps, loss/validation histories, gradient norms,
neighbor/self message ratios, occurrence changes, and residual scales. Ranking reports
include raw/filtered MRR, Hits@1/3/10, mean rank, head/tail and relation breakdowns,
owner-touching and non-owner slices, and unscorable counts. An owner slice is a useful
diagnostic, not a complete boundary on which predictions can benefit from structure.

```powershell
.venv/Scripts/python.exe -c "import torch,unittest; torch.set_num_threads(2); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

The supplied data's construction/selection limitations are unchanged. Historical scores
using structural edges as targets are not directly comparable to the corrected
ROOT-only objective. Preserve those reports as historical results.
