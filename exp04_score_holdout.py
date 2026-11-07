#!/usr/bin/env python3
"""EXP04: locked, offline, five-metric holdout scoring; no new Jitsi sessions.

Requires the UNCHANGED experiment plan, calibration lock, frozen scorer and
exp04_deep_calibration.py in D:\\visual-oracle-revision. Requires the 120 COMPLETE
holdout capture folders and original Y4M sources already on that computer.

Windows Anaconda Prompt:
    python exp04_score_holdout.py --check-only
    python exp04_score_holdout.py --max-new 2
    python exp04_score_holdout.py

The 120 holdout sessions (Parkrun and ScreenShare) are scored using the exact
calibration mask, normalization, preprocessing and model weights. The classifier
is called ONLY after all five-metric session summaries are finished. It uses
ONLY five session-level metrics; labels/conditions are used solely for audit.

No thresholds, model definitions, experiment data or existing capture files
are modified. Intermediate scores are checkpointed in a NEW folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np

import exp04_deep_calibration as deep

PLAN_SHA256 = '303dd62dfa915d729b4954bfca3469ecc025167136334d6afb895df27734732f'
LOCK_SHA256 = '26a06460a784c5079a28ac61d4e7caf27feb6c3c2383120f928d94c9ae52f725'
FROZEN_SHA256 = '12ab603c1305a5633f429f9051449832ea726d4ebd9cc2a4dcea0dfcc8dd1fe9'
DEEP_SCRIPT_SHA256 = 'b598ee50cdf4f96d53b44eeca3c54c0c105bd32323a2dc6bd97ce05e48a7a26c'
TARGET = (640, 360)
SAMPLE_FIELDS = ['job', 'video', 'block', 'condition', 'severity', 'sample',
                 'fault_active', 'frame_index', 'native_width', 'native_height',
                 'ssim', 'psnr_db', 'dists_distance', 'dists_similarity',
                 'dinov2_base_cosine', 'openclip_cosine']
SESSION_FIELDS = ['job', 'video', 'block', 'condition', 'severity', 'n', 'n_fault_active',
                  'mean_ssim', 'q10_ssim', 'mean_psnr_db', 'q10_psnr_db',
                  'mean_dists_distance', 'q90_dists_distance',
                  'mean_dino_cosine', 'q10_dinov2_base_cosine',
                  'mean_openclip_cosine', 'q10_openclip_cosine']


def require_hash(path, expected, description):
    if not path.is_file():
        raise FileNotFoundError(f'{description} not found: {path}')
    actual = deep.sha256_file(path)
    if actual != expected:
        raise RuntimeError(f'{description} SHA256 changed: {actual} != {expected}; STOP')


def check_lock(root):
    plan_path = root / 'captures/dashboard_jobs/EXP04_main240_plan.json'
    lock_path = root / 'captures/dashboard_jobs/EXP04_main240_calibration_lock.json'
    frozen_path = root / 'exp04_oracle_frozen.py'
    require_hash(plan_path, PLAN_SHA256, 'Plan')
    require_hash(lock_path, LOCK_SHA256, 'Calibration lock')
    require_hash(frozen_path, FROZEN_SHA256, 'Frozen classifier')
    require_hash(root / 'exp04_deep_calibration.py', DEEP_SCRIPT_SHA256,
                 'Previously used deep preprocessing/scoring module')
    lock = deep.read_json(lock_path)
    if lock.get('analysis_locked') is not True or lock.get('holdout_authorized') is not True:
        raise RuntimeError('Calibration lock is not active')
    if lock['scoring_protocol']['frozen_scorer_sha256'] != FROZEN_SHA256:
        raise RuntimeError('Frozen scorer not the one in the lock')
    if lock['calibration_videos'] != ['shields', 'fourpeople'] or lock['holdout_videos'] != ['parkrun', 'screenshare']:
        raise RuntimeError('Calibration/holdout video roles changed')
    return deep.read_json(plan_path), lock


def common_calibration_mask(root, plan, lock, q):
    masks = []
    for video in ('shields', 'fourpeople'):
        source = root / plan['videos'][video]['path']
        meta_path = source.with_name(source.stem + '_numbered_metadata.json')
        if not meta_path.is_file():
            raise FileNotFoundError(f'Calibration overlay metadata missing: {meta_path}')
        meta = deep.read_json(meta_path)
        if (int(meta['width']), int(meta['height'])) != (1280, 720):
            raise RuntimeError('Calibration barcode metadata dimensions changed')
        masks.append(q.valid_pixel_mask((360, 640, 3), meta)[0])
    mask = np.logical_and.reduce(masks)
    mask_sha = deep.sha256_bytes(np.ascontiguousarray(mask).tobytes())
    if mask_sha != lock['calibration_evidence']['common_mask_sha256']:
        raise RuntimeError(f'Calibration common ROI mask changed: {mask_sha}')
    return mask, mask_sha


def prepare_holdout(root, plan, lock, q):
    import exp04_main240 as core
    if Path(core.ROOT).resolve() != root.resolve():
        raise RuntimeError('exp04_main240.py is not loaded from project root')
    if core.PLAN.resolve() != (root / 'captures/dashboard_jobs/EXP04_main240_plan.json').resolve():
        raise RuntimeError('Unexpected experiment plan path')
    jobs = [j for j in plan['jobs'] if j['stage'] == 'holdout']
    if len(jobs) != 120 or {j['video'] for j in jobs} != {'parkrun', 'screenshare'}:
        raise RuntimeError('Holdout split not the planned 120 Parkrun/ScreenShare sessions')
    mask, mask_sha = common_calibration_mask(root, plan, lock, q)
    sources = {}
    for video in ('parkrun', 'screenshare'):
        path = root / plan['videos'][video]['path']
        require_hash(path, plan['videos'][video]['sha256'], f'{video} original Y4M')
        w, h, fps, offsets = q.index_y4m(path)
        if (w, h, len(offsets)) != (1280, 720, 500) or abs(fps-50) > 1e-6:
            raise RuntimeError(f'Original Y4M source dimensions/frames changed: {video}')
        sources[video] = {'path': path, 'w': w, 'h': h, 'offsets': offsets}

    prepared = {}
    exclusions = Counter()
    for j in jobs:
        status, state = core.job_status(j)
        if status != 'COMPLETE':
            raise RuntimeError(f'Holdout job not COMPLETE: {j["name"]} ({status})')
        run_folder = Path(state['results'][0]['run_folder']).resolve()
        try:
            run_folder.relative_to((root / 'captures').resolve())
        except ValueError as exc:
            raise RuntimeError(f'Run folder outside captures: {run_folder}') from exc
        csv_path = run_folder / 'fault_measurements.csv'
        raw = deep.read_csv(csv_path)
        if len(raw) != 31:
            raise RuntimeError(f'Expected 31 measurement records: {j["name"]}: {len(raw)}')
        selected, seen = [], set()
        for row in raw:
            sample = int(row['sample'])
            if sample in seen or not 0 <= sample < 31:
                raise RuntimeError(f'Duplicate or invalid sample number: {j["name"]}/{sample}')
            seen.add(sample)
            if row['quality_status'] != 'OK' or row['capture_status'] != 'OK':
                exclusions['quality_or_capture_not_OK'] += 1
                continue
            rw, rh = int(float(row['capture_width'])), int(float(row['capture_height']))
            if rw < TARGET[0] or rh < TARGET[1]:
                exclusions['below_640x360'] += 1
                continue
            if not row['frame_index'].isdigit() or not 0 <= int(row['frame_index']) < len(sources[j['video']]['offsets']):
                raise RuntimeError(f'Invalid decoded frame index: {j["name"]}/{sample}')
            filename = row['capture_filename']
            if (not filename or Path(filename).name != filename or
                    not filename.startswith('Participant_B_remote_') or not filename.endswith('.png')):
                raise RuntimeError(f'Invalid capture filename: {j["name"]}/{sample}/{filename}')
            png = run_folder / filename
            if not png.is_file():
                raise FileNotFoundError(f'PNG missing: {png}')
            if j['condition'] == 'baseline' and int(row['fault_active']) != 0:
                raise RuntimeError(f'Baseline marked fault-active: {j["name"]}/{sample}')
            selected.append(row)
        if len(seen) != 31:
            raise RuntimeError(f'Missing sample number in {j["name"]}')
        if len(selected) < int(lock['scoring_protocol']['minimum_eligible_frames_per_session']):
            raise RuntimeError(f'Only {len(selected)} eligible frames in {j["name"]}; STOP. '
                               'Locked protocol requires excluding this session and reporting it, not imputing.')
        prepared[j['name']] = {'job': j, 'folder': run_folder, 'rows': selected,
                               'input_sha256': deep.sha256_file(csv_path)}
    return jobs, sources, mask, mask_sha, prepared, exclusions


def score_structural(item, source, mask, q):
    result = {}
    with source['path'].open('rb') as f:
        for row in item['rows']:
            sample = int(row['sample'])
            received = cv2.imread(str(item['folder'] / row['capture_filename']), cv2.IMREAD_COLOR)
            native = (int(float(row['capture_width'])), int(float(row['capture_height'])))
            if received is None or received.shape[1::-1] != native:
                raise RuntimeError(f'PNG changed: {item["job"]["name"]}/{sample}')
            if native != TARGET:
                received = cv2.resize(received, TARGET, interpolation=cv2.INTER_AREA)
            frame = int(row['frame_index'])
            ref = q.read_reference(f, source['offsets'][frame], source['w'], source['h'], *TARGET)
            ssim, psnr = q.quality_metrics(ref, received, mask)
            if not math.isfinite(ssim) or not math.isfinite(psnr):
                raise RuntimeError(f'Nonfinite structural score: {item["job"]["name"]}/{sample}')
            result[sample] = (ssim, psnr)
    return result


def score_job(item, sources, mask, models, device, batch_size, ref_cache, q):
    deep_rows = deep.process_job(item, sources, mask, models, device, batch_size, ref_cache)
    structural = score_structural(item, sources[item['job']['video']], mask, q)
    rows = []
    for d in deep_rows:
        ssim, psnr = structural[int(d['sample'])]
        rows.append({**{k: d[k] for k in deep.SAMPLE_COLS},
                     'ssim': f'{ssim:.9f}', 'psnr_db': f'{psnr:.9f}'})
    if len(rows) != len(structural):
        raise RuntimeError('Deep/structural sample count mismatch')
    return rows


def validate_cache(path, meta_path, item, protocol_hash):
    meta = deep.read_json(meta_path)
    if (meta['input_sha256'] != item['input_sha256'] or
            meta['protocol_sha256'] != protocol_hash or
            meta['score_sha256'] != deep.sha256_file(path)):
        raise RuntimeError(f'Cached session/input/protocol changed: {path}')
    rows = deep.read_csv(path)
    expected = {int(x['sample']): x for x in item['rows']}
    if set(rows[0]) != set(SAMPLE_FIELDS) or len(rows) != len(expected):
        raise RuntimeError(f'Cached CSV schema/count invalid: {path}')
    if len({int(r['sample']) for r in rows}) != len(rows):
        raise RuntimeError(f'Cached sample duplicates: {path}')
    for r in rows:
        sample = int(r['sample'])
        if sample not in expected or r['job'] != item['job']['name']:
            raise RuntimeError(f'Cached sample mismatch: {path}')
        e = expected[sample]
        if int(r['frame_index']) != int(e['frame_index']) or int(r['fault_active']) != int(e['fault_active']):
            raise RuntimeError(f'Cached sample index/label mismatch: {path}')
        if (int(r['native_width']), int(r['native_height'])) != (
                int(float(e['capture_width'])), int(float(e['capture_height']))):
            raise RuntimeError(f'Cached dimensions mismatch: {path}')
        for metric in ('ssim', 'psnr_db', 'dists_distance', 'dinov2_base_cosine', 'openclip_cosine'):
            if not math.isfinite(float(r[metric])):
                raise RuntimeError(f'Cached nonfinite metric: {path}')
    return rows


def summarize(rows):
    d = deep.summarize_session(rows)
    a = np.array([float(x['ssim']) for x in rows], dtype=np.float64)
    b = np.array([float(x['psnr_db']) for x in rows], dtype=np.float64)
    return {'job': d['job'], 'video': d['video'], 'block': d['block'],
            'condition': d['condition'], 'severity': d['severity'], 'n': d['n'],
            'n_fault_active': d['n_fault_active'],
            'mean_ssim': float(a.mean()), 'q10_ssim': float(np.quantile(a, .1)),
            'mean_psnr_db': float(b.mean()), 'q10_psnr_db': float(np.quantile(b, .1)),
            'mean_dists_distance': d['mean_dists_distance'], 'q90_dists_distance': d['q90_dists_distance'],
            'mean_dino_cosine': d['mean_dinov2_base_cosine'],
            'q10_dinov2_base_cosine': d['q10_dinov2_base_cosine'],
            'mean_openclip_cosine': d['mean_openclip_cosine'],
            'q10_openclip_cosine': d['q10_openclip_cosine']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=r'D:\visual-oracle-revision')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-new', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 8 or (args.max_new is not None and args.max_new < 1):
        parser.error('Batch size must be 1..8; --max-new must be >= 1')
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Project root missing: {root}')
    os.chdir(root)
    sys.path.insert(0, str(root))
    print('[LOCK] Verifying original plan, active calibration lock and immutable scorer...', flush=True)
    plan, lock = check_lock(root)
    q = deep.import_quality(root)
    sys.modules['exp04_quality_offline'] = q
    jobs, sources, mask, mask_sha, prepared, excluded = prepare_holdout(root, plan, lock, q)
    total = sum(len(x['rows']) for x in prepared.values())
    print(f'[PREFLIGHT] PASS: 120/120 holdout COMPLETE; {total} eligible frames; '
          f'excluded={sum(excluded.values())} {dict(excluded)}; '
          f'common ROI {mask.mean()*100:.3f}%; mask SHA256={mask_sha}', flush=True)
    if args.check_only:
        print('[CHECK-ONLY] No models loaded; no experiment or output files changed.', flush=True)
        return

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    if device == 'auto':
        device = 'cpu'
    torch.manual_seed(20261106)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261106)
    torch.backends.cudnn.benchmark = False
    print(f'[MODELS] Loading on {device}; PyTorch {torch.__version__}', flush=True)
    models = deep.load_models(device)
    for key in ('dists_weights_sha256', 'dinov2_base_weights_sha256', 'openclip_weights_sha256'):
        if models[4][key] != lock['calibration_evidence'][key]:
            raise RuntimeError(f'Model weights changed since calibration: {key}; STOP')
    protocol = {
        'protocol': 'EXP04_holdout_5metric_frozen_v1',
        'plan_sha256': PLAN_SHA256, 'lock_sha256': LOCK_SHA256,
        'frozen_scorer_sha256': FROZEN_SHA256,
        'deep_calibration_script_sha256': DEEP_SCRIPT_SHA256,
        'resolution': [640, 360], 'no_upscaling': True,
        'common_mask_sha256': mask_sha, 'common_mask_eligible_fraction': float(mask.mean()),
        'preprocess': 'identical to exp04_deep_calibration.py; RGB(127,127,127) mask, full-frame 224x224 letterbox',
        'aggregation': 'q10 SSIM/PSNR/DINO/OpenCLIP; q90 DISTS; all eligible frames',
        'model_weights': models[4], 'device': device,
        'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__,
        'thresholds_tuned_on_holdout': False,
    }
    out = root / 'captures' / 'exp04_holdout_scoring_v1'
    job_dir = out / 'jobs'
    job_dir.mkdir(parents=True, exist_ok=True)
    proto_path = out / 'protocol.json'
    if proto_path.exists():
        if deep.read_json(proto_path) != protocol:
            raise RuntimeError('Holdout scoring protocol changed since checkpoint; STOP')
    else:
        deep.atomic_json(proto_path, protocol)
    protocol_hash = deep.sha256_file(proto_path)
    cached = {}
    for j in jobs:
        p = job_dir / (j['name'] + '.csv')
        m = job_dir / (j['name'] + '.json')
        if p.exists() != m.exists():
            raise RuntimeError(f'Incomplete checkpoint; do not overwrite: {p}')
        if p.exists():
            cached[j['name']] = validate_cache(p, m, prepared[j['name']], protocol_hash)
    print(f'[RESUME] {len(cached)}/120 holdout sessions already scored.', flush=True)
    ref_cache = {'dino': {}, 'clip': {}}
    newly = 0
    for idx, j in enumerate(jobs, 1):
        if j['name'] in cached:
            continue
        started = time.monotonic()
        item = prepared[j['name']]
        print(f'[{idx:03d}/120] {j["name"]} ({len(item["rows"])} eligible frames)', flush=True)
        rows = score_job(item, sources, mask, models, device, args.batch_size, ref_cache, q)
        p = job_dir / (j['name'] + '.csv')
        m = job_dir / (j['name'] + '.json')
        deep.write_csv_atomic(p, SAMPLE_FIELDS, rows)
        deep.atomic_json(m, {'input_sha256': item['input_sha256'],
                             'protocol_sha256': protocol_hash,
                             'score_sha256': deep.sha256_file(p)})
        cached[j['name']] = rows
        newly += 1
        print(f'    saved {len(rows)} rows in {time.monotonic()-started:.1f}s; '
              f'{len(cached)}/120 completed', flush=True)
        if args.max_new is not None and newly >= args.max_new:
            break
    if len(cached) != 120:
        print(f'[INCOMPLETE] {len(cached)}/120 scored; rerun without --max-new to resume.', flush=True)
        return

    samples = [r for j in jobs for r in cached[j['name']]]
    sessions = [summarize(cached[j['name']]) for j in jobs]
    if len(samples) != total or len(sessions) != 120:
        raise RuntimeError('Final holdout aggregation mismatch')
    samples_path = out / 'EXP04_holdout_5metric_samples.csv'
    sessions_path = out / 'EXP04_holdout_5metric_sessions.csv'
    deep.write_csv_atomic(samples_path, SAMPLE_FIELDS, samples)
    deep.write_csv_atomic(sessions_path, SESSION_FIELDS, sessions)
    # Frozen classifier itself checks the SHA256 of the plan, lock, and its own code.
    print('[EVALUATION] Applying the ORIGINAL frozen oracle; NO threshold tuning...', flush=True)
    subprocess.run([sys.executable, str(root / 'exp04_oracle_frozen.py'),
                    '--stage', 'holdout', '--input', str(sessions_path),
                    '--output-dir', str(out)], cwd=str(root), check=True)
    predictions_path = out / 'EXP04_holdout_frozen_predictions.csv'
    evaluation_path = out / 'EXP04_holdout_frozen_evaluation.json'
    audit = {
        'experiment': 'EXP04', 'stage': 'holdout',
        'completed_jobs': 120, 'expected_frames': 120*31,
        'eligible_frames': len(samples), 'excluded_frames': dict(excluded),
        'video_sessions': dict(Counter(j['video'] for j in jobs)),
        'condition_sessions': dict(Counter(j['condition'] for j in jobs)),
        'plan_sha256': PLAN_SHA256, 'calibration_lock_sha256': LOCK_SHA256,
        'frozen_scorer_sha256': FROZEN_SHA256,
        'deep_calibration_script_sha256': DEEP_SCRIPT_SHA256,
        'common_mask_sha256': mask_sha,
        'model_weights_sha256': models[4],
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'note': 'This is the first independent holdout evaluation; calibration thresholds unchanged.',
    }
    audit_path = out / 'EXP04_holdout_scoring_audit.json'
    deep.atomic_json(audit_path, audit)
    deliver = [samples_path, sessions_path, predictions_path, evaluation_path, audit_path, proto_path]
    checksums = {p.name: deep.sha256_file(p) for p in deliver}
    checksum_path = out / 'EXP04_holdout_checksums.json'
    deep.atomic_json(checksum_path, checksums)
    deliver.append(checksum_path)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive = root / 'captures' / f'EXP04_holdout_evaluation_{stamp}.zip'
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in deliver:
            z.write(p, p.name)
    print(f'\n[COMPLETE] 120 holdout sessions; {len(samples)} scored frames; '
          f'lock unchanged. UPLOAD: {archive}', flush=True)
    print('No Jitsi meeting started; no existing experiment or calibration files changed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
