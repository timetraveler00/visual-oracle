"""Read-only EXP01 post-hoc perceptual analysis (Windows / Anaconda Prompt).

Reads existing source Y4M, frame metadata, run CSVs and captured PNGs.
Writes ONLY under captures/exp01_perceptual_posthoc (or --output).
Never reruns Jitsi, changes original CSV/JSON, or resumes the completed job.

Usage:
  python exp01_perceptual_posthoc.py --check
  python exp01_perceptual_posthoc.py --limit 1
  python exp01_perceptual_posthoc.py
  python exp01_perceptual_posthoc.py --metrics DISTS DINOv2 OpenCLIP
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_JOB = ROOT / 'captures' / 'dashboard_jobs' / 'job_20261101_203713_195907'
DEFAULT_OUTPUT = ROOT / 'captures' / 'exp01_perceptual_posthoc'
METRICS = ('DISTS', 'DINOv2', 'DINOv2-Large', 'OpenCLIP')


def load_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def average(rows, key, active_only=False):
    values = []
    for row in rows:
        if row.get('quality_status') != 'OK':
            continue
        if active_only and str(row.get('fault_active')) not in ('1', '1.0', 'True'):
            continue
        try:
            x = float(row.get(key, ''))
            if math.isfinite(x):
                values.append(x)
        except (TypeError, ValueError):
            pass
    return round(sum(values) / len(values), 6) if values else ''


def prepare(root, job):
    cfg = json.loads((job / 'scenario.json').read_text(encoding='utf-8-sig'))
    status = json.loads((job / 'status.json').read_text(encoding='utf-8-sig'))
    if status.get('state') != 'completed':
        raise RuntimeError('Expected a completed EXP01 job; refusing to process incomplete job.')
    if cfg.get('name') != 'EXP01 – Blur Reproducibility':
        raise RuntimeError('Unexpected job name; inspect scenario.json before proceeding.')
    source = (root / cfg['video']).resolve()
    metadata_path = source.with_name(source.stem + '_numbered_metadata.json')
    if not source.is_file():
        raise FileNotFoundError(f'Missing original Y4M: {source}')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Missing barcode metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    results = status.get('results', [])
    if len(results) != 20:
        raise RuntimeError(f'Expected 20 completed runs, got {len(results)}')
    runs = []
    for i, result in enumerate(results, 1):
        # Only the final directory name is needed, so the script is portable.
        original = str(result.get('run_folder', '')).replace('\\', '/')
        dirname = original.rstrip('/').split('/')[-1]
        if not dirname.startswith('joint_'):
            raise RuntimeError(f'Unexpected run directory at {i}: {original}')
        folder = root / 'captures' / dirname
        path = folder / 'fault_measurements.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Run {i}: missing {path}')
        rows = load_csv(path)
        if not rows:
            raise RuntimeError(f'Run {i}: empty fault_measurements.csv')
        valid = [r for r in rows if r.get('quality_status') == 'OK']
        missing = [r.get('capture_filename', '') for r in valid
                   if not (folder / Path(r.get('capture_filename', '')).name).is_file()]
        if missing:
            raise FileNotFoundError(f'Run {i}: {len(missing)} missing captured PNGs; first: {missing[0]}')
        if len(valid) != int(result['quality_ok']):
            raise RuntimeError(f'Run {i}: quality_ok mismatch, CSV={len(valid)}, status={result["quality_ok"]}')
        runs.append({'index': i, 'result': result, 'folder': folder, 'rows': rows, 'valid': valid})
    return source, metadata_path, metadata, runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--job', type=Path, default=DEFAULT_JOB)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--check', action='store_true', help='Check input files without downloading models or writing files')
    p.add_argument('--limit', type=int, default=0, help='Process at most N runs (0=all), for pilot')
    p.add_argument('--metrics', nargs='+', choices=METRICS, default=list(METRICS))
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = p.parse_args()
    if args.limit < 0:
        p.error('--limit must be >= 0')
    job, output = args.job.resolve(), args.output.resolve()
    if output == job or job in output.parents or output == ROOT or output == ROOT / 'captures':
        p.error('Output must be a separate directory outside the original job')
    source, metadata_path, metadata, runs = prepare(ROOT, job)
    count = sum(len(r['valid']) for r in runs)
    print(f'[INPUT] Job: {job}')
    print(f'[INPUT] Source: {source}')
    print(f'[INPUT] Metadata: {metadata_path}')
    print(f'[INPUT] Runs: {len(runs)}, samples: {sum(len(r["rows"]) for r in runs)}, quality OK: {count}')
    print(f'[OUTPUT] Separate directory: {output}')
    if args.check:
        print('[CHECK] All original CSV files and quality-OK PNG files exist. No files changed.')
        return

    # Do not import or download large models until all input files have been checked.
    import cv2
    import numpy as np
    quality = load_module('quality08_exp01_posthoc', ROOT / '08_jitsi_synchronized_quality.py')
    perceptual = load_module('perceptual12_exp01_posthoc', ROOT / '12_perceptual_metrics.py')
    w, h, fps, offsets = quality.index_y4m(source)
    if len(offsets) != int(metadata['frame_count']):
        raise RuntimeError('Y4M frame count differs from barcode metadata')

    selected = list(dict.fromkeys(args.metrics))
    manifest_path = output / 'manifest.json'
    manifest = {'source': str(source), 'job': str(job), 'metrics': selected,
                'roi': 'crop below barcode overlay with 1% margin',
                'notes': 'DISTS=1-distance; DINOv2/OpenCLIP=cosine; ROI differs from masked SSIM'}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old != manifest:
            raise RuntimeError('Existing posthoc output uses a different source/job/metric set. Choose a new --output directory.')
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    # Initialize lazily: a resumed run should not reload large models if no work remains.
    chosen = runs[:args.limit] if args.limit else runs
    pending = []
    for run in chosen:
        path = output / f'run_{run["index"]:02d}_enhanced.csv'
        if path.exists():
            rows = load_csv(path)
            cols = [perceptual.METRIC_COLUMNS[m] for m in selected]
            if len(rows) != len(run['rows']) or any(any(not row.get(c, '') for c in cols)
                                                     for row in rows if row.get('quality_status') == 'OK'):
                raise RuntimeError(f'Incomplete existing output {path}. Inspect before retrying.')
            print(f'[SKIP] Run {run["index"]:02d}: previously completed')
        else:
            pending.append(run)

    engine = perceptual.PerceptualEngine(selected, device=args.device) if pending else None
    for run in pending:
        idx, folder = run['index'], run['folder']
        rows = [dict(row) for row in run['rows']]
        for row in rows:
            for m in selected:
                row[perceptual.METRIC_COLUMNS[m]] = ''
        with source.open('rb') as ref_file:
            for j, row in enumerate(rows, 1):
                if row.get('quality_status') != 'OK':
                    continue
                frame_idx = int(row['frame_index'])
                if not (0 <= frame_idx < len(offsets)):
                    raise RuntimeError(f'Run {idx}, sample {j}: frame index out of range: {frame_idx}')
                png = folder / Path(row['capture_filename']).name
                rec = cv2.imread(str(png), cv2.IMREAD_COLOR)
                if rec is None:
                    raise RuntimeError(f'Run {idx}, sample {j}: cannot decode {png}')
                rh, rw = rec.shape[:2]
                ref = quality.read_reference(ref_file, offsets[frame_idx], w, h, rw, rh)
                _, (_, _, _, y1) = quality.valid_pixel_mask(rec.shape, metadata)
                values = engine.compare(ref, rec, y1)
                for key, value in values.items():
                    row[key] = round(value, 6)
                print(f'[RUN {idx:02d}/20] frame {j:02d}/{len(rows)}: ' +
                      ' '.join(f'{k}={v:.4f}' for k, v in values.items()), flush=True)
        save_csv(output / f'run_{idx:02d}_enhanced.csv', rows, list(rows[0]))
        print(f'[SAVED] Run {idx:02d}', flush=True)

    # Build run-level summary from all completed post-hoc run files.
    summaries = []
    for run in runs:
        idx = run['index']
        path = output / f'run_{idx:02d}_enhanced.csv'
        if not path.is_file():
            continue
        rows = load_csv(path)
        result = run['result']
        item = {'plan_index': idx, 'repeat_index': result.get('repeat_index'),
                'scenario': result['scenario'], 'severity': result['severity'],
                'quality_ok': result['quality_ok'], 'sample_count': result['sample_count'],
                'mean_ssim_all': result.get('mean_ssim_all')}
        for m in selected:
            col = perceptual.METRIC_COLUMNS[m]
            item['mean_' + col + '_all'] = average(rows, col)
            item['mean_' + col + '_fault_active'] = average(rows, col, active_only=True)
        summaries.append(item)
    if summaries:
        fields = list(dict.fromkeys(k for row in summaries for k in row))
        save_csv(output / 'exp01_posthoc_summary.csv', summaries, fields)
    print(f'[DONE] Completed {len(summaries)}/20 run summaries. Output: {output}')
    if len(summaries) == 20:
        print('[DONE] All 20 runs enriched. Original experiment files were not modified.')
    else:
        print('[NOTE] Partial pilot complete. Run again without --limit to continue.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
