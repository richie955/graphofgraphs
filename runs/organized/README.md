# Saved GoG experiments

All collected local and Colab runs, organized on 2026-11-27.
Original reports, checkpoints, logs, diagnostics, and archive bytes are preserved.

See [RUN_DESCRIPTIONS.md](RUN_DESCRIPTIONS.md) for the purpose, dataset, initialization, configuration, and files for each run.

## Layout

- `local/seed_17/`: baseline80 pretraining, baseline80 + baseline80,
  baseline80 + GoG80, baseline80 + R-GCN80, and R-GCN160 from scratch.
- `colab/seed_17/`: baseline80, full-context GoG80, and ROOT-graph GoG80,
  each trained from scratch.
- `smoke/`: CPU comparison suite (including legacy GoG) and single-model CLI smoke.
  These are implementation checks, not full benchmarks.
- `support/`: original comparison manifest, dataset audit, Colab source bundle,
  and original Colab backup ZIP.
- `results.csv`: full-run results and current checkpoint/report locations.
- `artifact_manifest.json`: original-to-current file mapping, sizes, and SHA-256 hashes.

## Comparison protocol

All full runs use ROOT-only supervised targets. Full-context GoG also uses
internal and cross-context graph edges for message passing. The ROOT-graph
Colab experiment removes internal graph context; its dataset fingerprint differs.
The ROOT candidates and supervised query fingerprints match the full dataset.

Local warm-start branches share the same baseline80 checkpoint and reset their
optimizers before 80 additional updates. The baseline continuation is an 80+80
schedule, not uninterrupted 160-update training. R-GCN160 starts from scratch.
One epoch is one optimizer update. Checkpoints are selected by validation.
Only seed 17 has full runs, so these results do not establish consistent superiority.

## Saved results

| Run | Validation filtered MRR | Test filtered MRR |
|---|---:|---:|
| [colab/seed_17/baseline80_scratch](colab/seed_17/baseline80_scratch/baseline_report.json) | 0.162319 | 0.206367 |
| [colab/seed_17/gog80_full_context_scratch](colab/seed_17/gog80_full_context_scratch/gog_report.json) | 0.181406 | 0.214559 |
| [colab/seed_17/gog80_root_graph_scratch](colab/seed_17/gog80_root_graph_scratch/gog_report.json) | 0.172518 | 0.204855 |
| [local/seed_17/baseline80_plus_baseline80](local/seed_17/baseline80_plus_baseline80/report.json) | 0.189854 | 0.242616 |
| [local/seed_17/baseline80_plus_gog80](local/seed_17/baseline80_plus_gog80/gog_report.json) | 0.189623 | 0.241208 |
| [local/seed_17/baseline80_plus_rgcn80](local/seed_17/baseline80_plus_rgcn80/rgcn_report.json) | 0.183997 | 0.234396 |
| [local/seed_17/rgcn160_scratch](local/seed_17/rgcn160_scratch/report.json) | 0.200397 | 0.233521 |

The shared local baseline80 checkpoint is in
`local/seed_17/baseline80_pretraining/baseline_best.pt`.
No separate original local pretraining report was saved in the collected files.
No Colab R-GCN result was present in the supplied backup.

## Using the relocated files

Use checkpoint paths from `results.csv` for evaluation or further initialization.
Historical paths inside original reports/checkpoints still name their original
locations; these provenance records were intentionally preserved. Resolve old
paths using `artifact_manifest.json`. Colab `/content/...` paths were runtime paths.
The suite-level original manifest describes the intended suite; individual
saved reports determine which runs actually completed.

No models were retrained and no source datasets were changed during organization.
