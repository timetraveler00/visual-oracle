"""EXP04 smoke re-analysis at a fixed 640x360 evaluation scale.

Run from D:\\visual-oracle-revision after the FourPeople source verification:
    python exp04_analyze_smoke_640.py

Reads the TWO smoke job status files and the six existing Jitsi run folders.
Recomputes masked grayscale SSIM, masked RGB PSNR and DeltaE76 from the
original Y4M and receiver PNG at 640x360, using the project's own 08 quality
functions. Frames below 640x360 are UNSCORABLE_LOW_RES (never upscaled).
No new Jitsi session; no source/capture/job file is changed. Outputs are
written into a NEW timestamped directory under captures/exp04_smoke_640/.

IMPORTANT: Smoke data are for pipeline checks only, not a held-out evaluation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

JOBS = ('job_exp04_smoke_shields', 'job_exp04_smoke_fourpeople')
TARGET = (640, 360)


def load_quality(root: Path):
    path = root / '08_jitsi_synchronized_quality.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('exp04_quality08', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, col):
    vals = []
    for r in rows:
        v = r.get(col)
        try:
            v = float(v)
            if math.isfinite(v):
                vals.append(v)
        except (TypeError, ValueError):
            pass
    return round(statistics.mean(vals), 6) if vals else ''


def delta_e76(ref, rec, mask):
    a = cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(rec.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return float(np.mean(np.linalg.norm(a - b, axis=2)[mask]))


def resolve_run(root: Path, run_folder: str) -> Path:
    p = Path(run_folder)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    captures = (root / 'captures').resolve()
    if captures not in p.parents or not p.is_dir():
        raise ValueError(f'Run folder missing or outside captures: {p}')
    return p


def resolve_source(root: Path, job: dict) -> Path:
    raw = job.get('video')
    if not raw:
        raise ValueError('scenario.json has no video path')
    p = (root / raw).resolve()
    if p.parent != (root / 'videos').resolve() or not p.is_file() or p.suffix.lower() != '.y4m':
        raise ValueError(f'Invalid video source: {p}')
    return p


def run_analysis(root: Path, out: Path):
    quality = load_quality(root)
    samples = []
    summaries = []
    job_notes = []
    seen_runs = set()
    for job_name in JOBS:
        job_dir = root / 'captures' / 'dashboard_jobs' / job_name
        job = load_json(job_dir / 'scenario.json')
        status_file = job_dir / 'status.json'
        if not status_file.exists():
            status_file = job_dir / 'status_recovery.json'
        status = load_json(status_file)
        if status.get('state') != 'completed':
            raise RuntimeError(f'{job_name}: job is not completed ({status.get("state")})')
        source = resolve_source(root, job)
        meta = source.with_name(source.stem + '_numbered_metadata.json')
        metadata = load_json(meta)
        width, height, fps, offsets = quality.index_y4m(source)
        if (width, height, len(offsets)) != (int(metadata['width']), int(metadata['height']), int(metadata['frame_count'])):
            raise ValueError(f'Y4M/metadata mismatch: {source}')
        runs = status.get('results') or []
        if len(runs) != 3:
            raise ValueError(f'{job_name}: expected 3 smoke runs, got {len(runs)}')
        job_notes.append({'job': job_name, 'source': str(source), 'runs': len(runs)})
        with source.open('rb') as reference_file:
            for run_info in runs:
                run_dir = resolve_run(root, run_info['run_folder'])
                if run_dir in seen_runs:
                    raise ValueError(f'Duplicate run folder: {run_dir}')
                seen_runs.add(run_dir)
                run_settings = load_json(run_dir / 'settings.json')
                if Path(str(run_settings.get('reference_video', ''))).name != source.name:
                    raise ValueError(f'Wrong source in run settings: {run_dir}')
                fault_meta = run_settings.get('fault_injection') or {}
                condition = run_info.get('scenario') or fault_meta.get('scenario')
                severity = run_info.get('severity') or fault_meta.get('severity')
                if condition not in {'baseline', 'blur', 'freeze'}:
                    raise ValueError(f'Unexpected smoke condition {condition} in {run_dir}')
                if fault_meta.get('scenario') != condition:
                    raise ValueError(f'Condition mismatch in settings: {run_dir}')
                if run_info.get('run_valid') is False:
                    raise RuntimeError(f'Invalid stream continuity: {run_dir}')
                start = fault_meta.get('fault_start_frame')
                end = fault_meta.get('fault_end_frame')
                rows = read_csv(run_dir / 'fault_measurements.csv')
                if len(rows) != 31:
                    raise ValueError(f'Expected 31 samples, got {len(rows)} in {run_dir}')
                session = []
                for r in rows:
                    result = {
                        'job': job_name, 'video': source.stem, 'run_folder': str(run_dir),
                        'condition': condition, 'severity': severity,
                        'sample': r.get('sample', ''), 'capture_filename': r.get('capture_filename', ''),
                        'native_width': r.get('capture_width', ''), 'native_height': r.get('capture_height', ''),
                        'frame_index': r.get('frame_index', ''), 'fault_active': '',
                        'status_640': '', 'ssim_640': '', 'psnr_db_640': '',
                        'delta_e76_640': '', 'error': '',
                    }
                    try:
                        if r.get('quality_status') != 'OK':
                            result['status_640'] = 'UNSCORABLE_CAPTURE_ERROR'
                            result['error'] = r.get('quality_error', '') or r.get('capture_error', '')
                        else:
                            capture = run_dir / r['capture_filename']
                            if not capture.is_file():
                                raise FileNotFoundError(f'PNG missing: {capture}')
                            received = cv2.imread(str(capture), cv2.IMREAD_COLOR)
                            if received is None:
                                raise ValueError('PNG cannot be decoded')
                            rh, rw = received.shape[:2]
                            result['native_width'], result['native_height'] = rw, rh
                            if rw < TARGET[0] or rh < TARGET[1]:
                                result['status_640'] = 'UNSCORABLE_LOW_RES'
                            else:
                                idx, _, _ = quality.decode_barcode(received, metadata)
                                if not 0 <= idx < len(offsets):
                                    raise ValueError(f'Barcode out of source range: {idx}')
                                if idx != int(r['frame_index']):
                                    raise ValueError(f'Barcode mismatch: CSV {r["frame_index"]}, PNG {idx}')
                                result['frame_index'] = idx
                                expected_active = int(condition != 'baseline' and start is not None and end is not None
                                                      and int(start) <= idx <= int(end))
                                if r.get('fault_active', '') not in ('', str(expected_active)):
                                    raise ValueError(f'Fault label mismatch: CSV={r["fault_active"]}, expected={expected_active}')
                                result['fault_active'] = expected_active
                                resized = (cv2.resize(received, TARGET, interpolation=cv2.INTER_AREA)
                                           if (rw, rh) != TARGET else received)
                                reference = quality.read_reference(reference_file, offsets[idx], width, height, *TARGET)
                                mask, _ = quality.valid_pixel_mask(resized.shape, metadata)
                                ssim, psnr = quality.quality_metrics(reference, resized, mask)
                                result['ssim_640'] = round(ssim, 6)
                                result['psnr_db_640'] = round(psnr, 6) if math.isfinite(psnr) else 'inf'
                                result['delta_e76_640'] = round(delta_e76(reference, resized, mask), 6)
                                result['status_640'] = 'OK'
                    except Exception as e:
                        result['status_640'] = 'ERROR'
                        result['error'] = str(e)
                    samples.append(result)
                    session.append(result)
                valid = [x for x in session if x['status_640'] == 'OK']
                active = [x for x in valid if x['fault_active'] == 1]
                inactive = [x for x in valid if x['fault_active'] == 0]
                summary = {
                    'job': job_name, 'video': source.stem, 'condition': condition,
                    'severity': severity, 'run_folder': str(run_dir),
                    'total_samples': len(session), 'valid_640': len(valid),
                    'low_res_excluded': sum(x['status_640'] == 'UNSCORABLE_LOW_RES' for x in session),
                    'other_unscorable': sum(x['status_640'] not in ('OK', 'UNSCORABLE_LOW_RES') for x in session),
                    'coverage_pct': round(100 * len(valid) / len(session), 2),
                    'active_valid': len(active), 'inactive_valid': len(inactive),
                    'ssim_all_640': mean(valid, 'ssim_640'),
                    'ssim_active_640': mean(active, 'ssim_640'),
                    'ssim_inactive_640': mean(inactive, 'ssim_640'),
                    'psnr_all_db_640': mean(valid, 'psnr_db_640'),
                    'delta_e76_all_640': mean(valid, 'delta_e76_640'),
                }
                summaries.append(summary)
                print(f'{source.stem:24} {condition:8} {severity:6} '
                      f'{len(valid):2}/{len(session)}  SSIM={summary["ssim_all_640"]}', flush=True)

    out.mkdir(parents=True, exist_ok=False)
    sample_cols = ['job','video','run_folder','condition','severity','sample','capture_filename',
                   'native_width','native_height','frame_index','fault_active','status_640',
                   'ssim_640','psnr_db_640','delta_e76_640','error']
    run_cols = list(summaries[0])
    save_csv(out / 'EXP04_smoke_640_samples.csv', samples, sample_cols)
    save_csv(out / 'EXP04_smoke_640_runs.csv', summaries, run_cols)
    manifest = {
        'purpose': 'smoke pipeline re-analysis, not independent test',
        'scale': '640x360', 'upscaling': 'prohibited',
        'metric_implementation': '08_jitsi_synchronized_quality.py (masked grayscale SSIM, masked RGB PSNR)',
        'delta_e76': 'OpenCV float32 Lab, masked mean pixel distance',
        'jobs': job_notes,
        'runs': len(summaries), 'samples': len(samples),
        'valid_samples': sum(x['status_640'] == 'OK' for x in samples),
        'low_res_excluded': sum(x['status_640'] == 'UNSCORABLE_LOW_RES' for x in samples),
        'other_unscorable': sum(x['status_640'] not in ('OK', 'UNSCORABLE_LOW_RES') for x in samples),
    }
    (out / 'EXP04_smoke_640_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nOutput:', out)
    print('Sessions:', len(summaries), '| samples:', len(samples),
          '| valid:', manifest['valid_samples'],
          '| low-res excluded:', manifest['low_res_excluded'],
          '| other unscorable:', manifest['other_unscorable'])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'captures' / 'exp04_smoke_640'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = base / stamp
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    return run_analysis(root, out)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(2)
