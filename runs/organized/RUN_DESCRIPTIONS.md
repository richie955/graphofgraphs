# What each saved run used

This guide describes the saved artifacts, not a proposed experiment. Original JSON reports remain unchanged.

## Shared full-run protocol

All seven full-run reports use seed 17, CUDA, hidden dimension 128, 50 negatives per side, decoder fact batch size 4096, learning rate 0.001, weight decay 0.0001, dropout 0.2, and gradient clipping 1.0. Graph models use eight bases; GoG attention uses four heads. Graph-only settings recorded in a baseline config are not active baseline components.

Each epoch performs ONE optimizer update. Fact batching divides the loss calculation; it does not multiply updates. Runs use fixed update budgets and select the checkpoint with the best validation filtered MRR. Phase-two starting checkpoints remain eligible.

The full dataset is `structural_gog/gog/gog_model_subset_coarse.json`: 16,462 supervised ROOT training facts, 2,000 validation facts, 2,007 test facts, and 33,742 training facts available as graph input. The ROOT-graph variant is `gog_model_subset_rootonly.json`: the same supervised ROOT splits, but only 16,462 ROOT training facts as graph input. These counts come from the saved dataset audit. The larger balanced dataset was inspected but has no full training run in this collection.

ROOT-only task scope means only ROOT facts become training targets. It does NOT mean the internal graph was removed. Full-context GoG retains local and cross-context messages. All ROOT evaluations rank against 2,534 ROOT candidates, with both head and tail prediction: 4,000 validation queries and 4,014 test queries. ComplEx scoring and existing filtering rules are shared. Validation filtering uses train+validation facts; test filtering uses train+validation+test facts.

The structural GoG runs use categorical initialization, the identity-preserving ROOT residual, local and ROOT propagation, sparse cross-context attention, family readout, and upward fusion. They use one round, with synchronization and feedback disabled. Null message settings in the reports select the structural defaults; they are not missing measurements. Baseline and R-GCN use their own architectures; GoG stage switches in their shared configuration do not activate GoG stages.

## Shared local pretraining checkpoint

`local/seed_17/baseline80_pretraining/baseline_best.pt` is the entity-only baseline trained for 80 updates, selected at epoch 80. All three local continuation runs copy this initializer and ComplEx decoder by ID and reset their optimizers. There is no separate saved pretraining JSON report in this collection. The initialization records in the continuation reports identify the source checkpoint.

## Full runs

### colab/seed_17/baseline80_scratch

Entity-only baseline, randomly initialized, trained for 80 updates on the full dataset. No graph propagation. This is the Colab reference for the two scratch GoG runs.

- Report: [baseline_report.json](colab/seed_17/baseline80_scratch/baseline_report.json).
- Weights: [baseline_best.pt](colab/seed_17/baseline80_scratch/baseline_best.pt).
- Completed phase updates: 80; selected phase epoch: 80.
- Validation filtered MRR: 0.162319; test filtered MRR: 0.206367.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": null,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 80,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "legacy",
  "task_scope": "root",
  "feature_mode": "auto",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": true,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": true,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": null,
  "early_stopping": false,
  "update_offset": 0,
  "evaluate_test": true,
  "diagnostic_train_limit": 32
}
```

### colab/seed_17/gog80_full_context_scratch

New structural GoG, randomly initialized, trained for 80 updates on the full dataset. ROOT supervision with internal/cross-context graph input. Its training report has test=null because test evaluation was run later; the saved gog_test_evaluation.json contains those test metrics.

- Report: [gog_report.json](colab/seed_17/gog80_full_context_scratch/gog_report.json).
- Weights: [gog_best.pt](colab/seed_17/gog80_full_context_scratch/gog_best.pt).
- Completed phase updates: 80; selected phase epoch: 79.
- Validation filtered MRR: 0.181406; test filtered MRR: 0.214559.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": 1,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 80,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "structural",
  "task_scope": "root",
  "feature_mode": "categorical",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": true,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": true,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": null,
  "early_stopping": false,
  "update_offset": 0,
  "evaluate_test": false,
  "diagnostic_train_limit": 32
}
```

### colab/seed_17/gog80_root_graph_scratch

New structural GoG, randomly initialized, trained for 80 updates on the ROOT-graph dataset. Same ROOT supervision, candidates, and query splits, but internal context is absent. Enabled local/readout switches cannot supply internal information when that graph is absent.

- Report: [gog_report.json](colab/seed_17/gog80_root_graph_scratch/gog_report.json).
- Weights: [gog_best.pt](colab/seed_17/gog80_root_graph_scratch/gog_best.pt).
- Completed phase updates: 80; selected phase epoch: 80.
- Validation filtered MRR: 0.172518; test filtered MRR: 0.204855.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": 1,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 80,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "structural",
  "task_scope": "root",
  "feature_mode": "categorical",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": true,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": true,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": null,
  "early_stopping": false,
  "update_offset": 0,
  "evaluate_test": true,
  "diagnostic_train_limit": 32
}
```

### local/seed_17/baseline80_plus_baseline80

Entity-only baseline continued for 80 updates from the shared baseline80 checkpoint. Total training budget is 160 updates with an optimizer reset at the phase boundary. This is not uninterrupted baseline160 training. Test metrics were evaluated afterward and saved separately in test_results.json.

- Report: [report.json](local/seed_17/baseline80_plus_baseline80/report.json).
- Weights: [baseline_best.pt](local/seed_17/baseline80_plus_baseline80/baseline_best.pt).
- Completed phase updates: 80; selected phase epoch: 48; selected cumulative update: 128.
- Validation filtered MRR: 0.189854; test filtered MRR: 0.242616.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": 2,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 80,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "legacy",
  "task_scope": "root",
  "feature_mode": "id_type",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": false,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": false,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": "C:\\Users\\richi\\Desktop\\Graph of Graphs\\runs\\structural-real-seed17\\seed_17\\pretraining\\baseline_best.pt",
  "early_stopping": false,
  "update_offset": 80,
  "evaluate_test": false,
  "diagnostic_train_limit": 32
}
```

### local/seed_17/baseline80_plus_gog80

New structural GoG initialized from the shared baseline80 checkpoint, then trained for 80 updates on the full dataset. Graph parameters start newly initialized; the ROOT residual begins at zero. Total shared pretraining plus continuation budget is 160 updates. The saved validation_graph_contribution.json is an inference-only validation diagnostic comparing the learned graph residual enabled versus disabled; it is not another training run.

- Report: [gog_report.json](local/seed_17/baseline80_plus_gog80/gog_report.json).
- Weights: [gog_best.pt](local/seed_17/baseline80_plus_gog80/gog_best.pt).
- Completed phase updates: 80; selected phase epoch: 47; selected cumulative update: 127.
- Validation filtered MRR: 0.189623; test filtered MRR: 0.241208.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": 1,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 80,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "structural",
  "task_scope": "root",
  "feature_mode": "categorical",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": true,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": true,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": "C:\\Users\\richi\\Desktop\\Graph of Graphs\\runs\\structural-real-seed17\\seed_17\\pretraining\\baseline_best.pt",
  "early_stopping": false,
  "update_offset": 80,
  "evaluate_test": true,
  "diagnostic_train_limit": 32
}
```

### local/seed_17/baseline80_plus_rgcn80

R-GCN initialized from the shared baseline80 initializer and decoder, then trained for 80 updates on the full dataset. Graph layers start newly initialized. Total budget including shared pretraining is 160 updates. This model does not use the structural GoG residual or hierarchy.

- Report: [rgcn_report.json](local/seed_17/baseline80_plus_rgcn80/rgcn_report.json).
- Weights: [rgcn_best.pt](local/seed_17/baseline80_plus_rgcn80/rgcn_best.pt).
- Completed phase updates: 80; selected phase epoch: 80; selected cumulative update: 160.
- Validation filtered MRR: 0.183997; test filtered MRR: 0.234396.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": null,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 80,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "legacy",
  "task_scope": "root",
  "feature_mode": "auto",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": true,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": true,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": "C:\\Users\\richi\\Desktop\\Graph of Graphs\\runs\\structural-real-seed17\\seed_17\\pretraining\\baseline_best.pt",
  "early_stopping": false,
  "update_offset": 80,
  "evaluate_test": true,
  "diagnostic_train_limit": 32
}
```

### local/seed_17/rgcn160_scratch

R-GCN randomly initialized and trained for 160 updates on the full dataset. It does not start from baseline80. Test metrics were evaluated afterward and saved separately in test_results.json.

- Report: [report.json](local/seed_17/rgcn160_scratch/report.json).
- Weights: [rgcn_best.pt](local/seed_17/rgcn160_scratch/rgcn_best.pt).
- Completed phase updates: 160; selected phase epoch: 158.
- Validation filtered MRR: 0.200397; test filtered MRR: 0.233521.
- Recorded configuration:

```json
{
  "hidden_dim": 128,
  "num_bases": 8,
  "num_heads": 4,
  "dropout": 0.2,
  "num_rounds": 2,
  "negatives_per_side": 50,
  "training_batch_size": 4096,
  "epochs": 160,
  "patience": 15,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip": 1.0,
  "seed": 17,
  "device": "cuda",
  "show_progress": true,
  "encoder_profile": "legacy",
  "task_scope": "root",
  "feature_mode": "id_type",
  "basis_init": null,
  "composition": null,
  "normalization": null,
  "identity_residual": false,
  "local_enabled": true,
  "cross_enabled": true,
  "fusion_enabled": true,
  "root_enabled": true,
  "sync_enabled": false,
  "feedback_enabled": false,
  "family_readout": false,
  "embedding_learning_rate": null,
  "encoder_learning_rate": null,
  "decoder_learning_rate": null,
  "init_checkpoint": null,
  "early_stopping": false,
  "update_offset": 0,
  "evaluate_test": false,
  "diagnostic_train_limit": 32
}
```

## Smoke reports

`smoke/comparison_suite/` is a tiny CPU fixture comparison: hidden dimension 16, two bases, four attention heads, two negatives per side, fact batch size four, seed 17, and only two supervised ROOT training targets. Baseline pretraining runs for two updates; baseline continuation and structural GoG run two more; R-GCN and legacy GoG run four from scratch. Each model directory contains its own `report.json` and best checkpoint. The structural model uses one effective round; legacy GoG uses two. No test evaluation was requested. `comparison.csv`, `comparison.json`, and `manifest.json` describe the suite. Scores are correctness diagnostics, not real-dataset benchmarks.

`smoke/single_model_cli/gog_report.json` is a separate two-update, hidden-16 structural GoG CPU check on the tiny fixture, starting from random weights. Its purpose is to verify the single-model training command and checkpoint output.

## How to read the files

- `baseline_report.json`, `gog_report.json`, `rgcn_report.json`: CLI report wrappers with `config`, split counts, and model results under `baseline`, `gog`, or `rgcn`.
- `report.json`: suite result, with configuration and metrics directly at the top level.
- `config`: requested settings. Some shared fields are unused by the chosen architecture. Null override fields use model defaults.
- `loss_history`: training loss per optimizer update, including updates after the selected checkpoint.
- `best_epoch`: selected phase checkpoint; it may be earlier than the completed budget. `update_offset` records preceding shared baseline updates.
- `initialization`: random versus transferred weights, source checkpoint, ID-copy counts, and optimizer reset status.
- `fingerprints`: dataset structure, candidate list, and query identities. `negative_sample_fingerprints` records sampled negatives per update.
- `validation` and `test`: raw/filtered MRR, mean rank, Hits@1/3/10, plus relation, prediction-side, and owner slices where recorded. Null test metrics mean test was not evaluated by that training invocation; check the separate evaluation file.
- `diagnostics`: recorded gradient norms, message/self ratios, residual scales, and diagnostic ranking where available.
- `test_results.json` and `gog_test_evaluation.json`: evaluation of existing selected weights, with no retraining.
- `training.log`: console progress, available for the Colab full-context GoG run.
- `support/structural-data-audit.json`: data loading, finite-value, and source-file preservation checks; not a training result.
- `support/local_comparison_manifest.json`: intended original suite configuration, not proof that every planned model completed.
- `artifact_manifest.json`: original/current file locations and integrity hashes. Old paths inside reports/checkpoints remain historical provenance; use the current paths in `results.csv`.

Only seed 17 has real training runs. No full legacy-GoG run or Colab R-GCN run is present. These results do not yet cover multiple seeds or all targeted ablations.
