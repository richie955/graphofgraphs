"""One-time organization of saved experiments; preserves artifact bytes."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'runs/organized'
LOCAL = 'runs/structural-real-seed17/seed_17/'
COLAB = 'structural_gog/gog/gog_runs_backup/gog_runs/'
MOVES = [
    (LOCAL + 'pretraining', 'local/seed_17/baseline80_pretraining'),
    (LOCAL + 'baseline_continuation', 'local/seed_17/baseline80_plus_baseline80'),
    (LOCAL + 'rgcn', 'local/seed_17/rgcn160_scratch'),
    (LOCAL + 'gog_structural_warm80_20261127_040630', 'local/seed_17/baseline80_plus_gog80'),
    (LOCAL + 'rgcn_warm80_20261127_041939', 'local/seed_17/baseline80_plus_rgcn80'),
    (COLAB + 'baseline_80_20261126_221304', 'colab/seed_17/baseline80_scratch'),
    (COLAB + 'gog_only_80_20261126_215709', 'colab/seed_17/gog80_full_context_scratch'),
    (COLAB + 'rootonly_scratch80_20261126_224539', 'colab/seed_17/gog80_root_graph_scratch'),
    ('runs/structural-smoke', 'smoke/comparison_suite'),
    ('runs/structural-cli-smoke', 'smoke/single_model_cli'),
    ('runs/structural-real-seed17/manifest.json', 'support/local_comparison_manifest.json'),
    ('runs/structural-data-audit.json', 'support/structural-data-audit.json'),
    ('runs/colab/gog_embedding_updated.zip', 'support/colab_source/gog_embedding_updated.zip'),
    ('structural_gog/gog/gog_runs_backup.zip', 'support/original_archives/gog_runs_backup.zip'),
]


def checked(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError(f'Outside intended workspace: {resolved}')
    return resolved


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


if DEST.exists():
    raise SystemExit(f'Refusing to overwrite {DEST}')

# Resolve and validate every source and destination before moving anything.
plan = [(checked(ROOT / a), checked(DEST / b)) for a, b in MOVES]
inventory = []
for src, dst in plan:
    if not src.exists() or dst.exists():
        raise RuntimeError(f'Missing source or existing destination: {src} -> {dst}')
    for file in sorted(src.rglob('*')) if src.is_dir() else [src]:
        if file.is_file():
            target = dst / file.relative_to(src) if src.is_dir() else dst
            inventory.append({'original_path': file.relative_to(ROOT).as_posix(),
                              'organized_path': target.relative_to(DEST).as_posix(),
                              'bytes': file.stat().st_size, 'sha256': digest(file)})

DEST.mkdir(parents=True)
(DEST / 'artifact_manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
for src, dst in plan:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
for item in inventory:
    file = DEST / item['organized_path']
    assert file.stat().st_size == item['bytes'] and digest(file) == item['sha256'], file

rows = []
for report in sorted(DEST.rglob('*report.json')):
    if 'smoke' in report.relative_to(DEST).parts:
        continue
    data = json.loads(report.read_text())
    result = next((v for v in data.values() if isinstance(v, dict) and 'best_epoch' in v), data)
    validation = result.get('validation') or {}
    test = result.get('test') or {}
    if not test and (report.parent / 'test_results.json').exists():
        test = json.loads((report.parent / 'test_results.json').read_text())['test']
    if not test and (report.parent / 'gog_test_evaluation.json').exists():
        test = json.loads((report.parent / 'gog_test_evaluation.json').read_text())
    config = data.get('config', {})
    checkpoint = next(report.parent.glob('*.pt'))
    row = {'run': report.parent.relative_to(DEST).as_posix(),
           'seed': config.get('seed'), 'task_scope': config.get('task_scope'),
           'phase_updates': result.get('optimizer_updates'),
           'pretraining_updates': result.get('update_offset', 0),
           'selected_phase_epoch': result.get('best_epoch'),
           'validation_filtered_mrr': validation.get('root', {}).get('filtered', {}).get('mrr'),
           'test_filtered_mrr': test.get('root', {}).get('filtered', {}).get('mrr'),
           'checkpoint': checkpoint.relative_to(DEST).as_posix(),
           'report': report.relative_to(DEST).as_posix(),
           'dataset_fingerprint': result.get('fingerprints', {}).get('dataset')}
    rows.append(row)
with (DEST / 'results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

table = ['| Run | Validation filtered MRR | Test filtered MRR |',
         '|---|---:|---:|']
for row in rows:
    fmt = lambda value: f'{value:.6f}' if value is not None else 'Not saved'
    table.append(f"| [{row['run']}]({row['report']}) | {fmt(row['validation_filtered_mrr'])} | {fmt(row['test_filtered_mrr'])} |")
readme = '''# Saved GoG experiments

All collected local and Colab runs, organized on 2026-11-27.
Original reports, checkpoints, logs, diagnostics, and archive bytes are preserved.

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

'''
readme += '\n'.join(table)
readme += '''

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
'''
(DEST / 'README.md').write_text(readme, encoding='utf-8')
print(f'Organized and SHA-256 verified {len(inventory)} original files in {DEST}')
print(f'Indexed {len(rows)} full-run reports plus the shared pretraining checkpoint.')
