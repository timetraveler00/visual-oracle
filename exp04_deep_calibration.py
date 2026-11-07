#!/usr/bin/env python3
"""EXP04: offline DISTS / DINOv2-Base / OpenCLIP calibration on 120 existing Jitsi sessions.

Run from D:\\visual-oracle-revision with the existing visual-oracle Python environment.
No Jitsi, Docker, source modifications, or holdout access. Checkpoints per session.

    python exp04_deep_calibration.py --check-only
    python exp04_deep_calibration.py --max-new 2
    python exp04_deep_calibration.py

Produces captures/EXP04_deep_calibration_*.zip only after all 120 sessions finish.

Scientific protocol:
 - Evaluation unit: independent Jitsi session (not individual PNG).
 - Eligible frames: exactly the 3665 samples previously selected for offline 640x360
   structural analysis; low-res frames are not upscaled.
 - Reference: clean Y4M frame indexed by receiver barcode; no temporal alignment guesses.
 - Mask: intersection of the two calibration videos' barcode overlay exclusion masks.
 - For deep models, excluded pixels in BOTH images are replaced with identical neutral
   RGB (127,127,127) pixels; full 640x360 frame retained for DISTS.
 - DINOv2 / OpenCLIP input: entire 640x360 image letterboxed to 224x224 (no crop).
 - DISTS is a distance (larger = worse); DINOv2 and OpenCLIP are cosine similarities
   (smaller = worse). No thresholds are chosen by this script.
 - Frame-level fault_active is saved for descriptive diagnostics ONLY, never used as
   a classifier input or as a frame-selection criterion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image, ImageOps

PLAN_SHA256 = '303dd62dfa915d729b4954bfca3469ecc025167136334d6afb895df27734732f'
SAMPLES_SHA256 = '229c6ccda37f98f46929b178a33e1f41d2d742e832959d0ab4957b2c1c4fc2a9'
DINO_REPO = 'facebookresearch/dinov2'
DINO_NAME = 'dinov2_vitb14'
OPENCLIP_NAME = 'ViT-B-32'
OPENCLIP_WEIGHTS = 'laion2b_s34b_b79k'
TARGET = (640, 360)
DEEP_SIZE = (224, 224)
NEUTRAL_RGB = (127, 127, 127)
SAMPLE_COLS = ['job', 'video', 'block', 'condition', 'severity', 'sample',
               'fault_active', 'frame_index', 'native_width', 'native_height',
               'dists_distance', 'dists_similarity', 'dinov2_base_cosine',
               'openclip_cosine']
SESSION_COLS = ['job', 'video', 'block', 'condition', 'severity', 'n', 'n_fault_active',
                'mean_dists_distance', 'q90_dists_distance', 'mean_dists_similarity',
                'q10_dists_similarity', 'mean_dinov2_base_cosine',
                'q10_dinov2_base_cosine', 'mean_openclip_cosine', 'q10_openclip_cosine']


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv_atomic(path, fields, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def import_quality(root):
    path = root / '08_jitsi_synchronized_quality.py'
    if not path.is_file():
        raise FileNotFoundError(f'Quality module missing: {path}')
    spec = importlib.util.spec_from_file_location('exp04_quality_offline', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_calibration(root):
    plan_path = root / 'captures' / 'dashboard_jobs' / 'EXP04_main240_plan.json'
    samples_path = root / 'captures' / 'EXP04_calibration_640_samples.csv'
    for p, expected in [(plan_path, PLAN_SHA256), (samples_path, SAMPLES_SHA256)]:
        if not p.is_file():
            raise FileNotFoundError(f'Required existing file missing: {p}')
        actual = sha256_file(p)
        if actual != expected:
            raise RuntimeError(f'Input SHA256 mismatch for {p}: {actual} != {expected}')
    plan = read_json(plan_path)
    assert plan.get('experiment') == 'EXP04' and len(plan['jobs']) == 240
    jobs = [j for j in plan['jobs'] if j['stage'] == 'calibration']
    assert len(jobs) == 120
    assert {j['video'] for j in jobs} == {'shields', 'fourpeople'}
    assert {j['video'] for j in plan['jobs'] if j['stage'] == 'holdout'} == {'parkrun', 'screenshare'}

    selected = read_csv(samples_path)
    assert len(selected) == 3665, f'Expected 3665 selected frames; got {len(selected)}'
    whitelist = defaultdict(dict)
    for row in selected:
        name, sample = row['job'], int(row['sample'])
        if sample in whitelist[name]:
            raise ValueError(f'Duplicate selected sample: {name}/{sample}')
        if int(row['native_width']) < 640 or int(row['native_height']) < 360:
            raise ValueError(f'Unexpected upscaled frame in selection: {name}/{sample}')
        whitelist[name][sample] = row
    if set(whitelist) != {j['name'] for j in jobs}:
        raise ValueError('Selected CSV does not contain exactly the 120 calibration jobs')

    import exp04_main240 as core
    if Path(core.ROOT).resolve() != root.resolve():
        raise RuntimeError('exp04_main240.py must be located in the project root')
    manifest = core.load_plan()
    if sha256_file(core.PLAN) != PLAN_SHA256:
        raise RuntimeError('Main runner plan hash mismatch')
    if len(manifest['jobs']) != 240:
        raise RuntimeError('Main runner plan count mismatch')

    prepared = {}
    for j in jobs:
        status, state = core.job_status(j)
        if status != 'COMPLETE':
            raise RuntimeError(f'Calibration session not COMPLETE: {j["name"]} ({status})')
        folder = Path(state['results'][0]['run_folder'])
        csv_path = folder / 'fault_measurements.csv'
        rows = read_csv(csv_path)
        if len(rows) != 31:
            raise ValueError(f'Expected 31 samples for {j["name"]}, got {len(rows)}')
        eligible = []
        for row in rows:
            sample = int(row['sample'])
            saved = whitelist[j['name']].get(sample)
            if saved is None:
                continue
            if row['quality_status'] != 'OK' or row['capture_status'] != 'OK':
                raise ValueError(f'Invalid selected frame status: {j["name"]}/{sample}')
            if (int(float(row['capture_width'])), int(float(row['capture_height']))) != (
                    int(saved['native_width']), int(saved['native_height'])):
                raise ValueError(f'Resolution mismatch: {j["name"]}/{sample}')
            if row['fault_active'] != saved['fault_active']:
                raise ValueError(f'Fault-active mismatch: {j["name"]}/{sample}')
            fname = row['capture_filename']
            if not fname or Path(fname).name != fname or not fname.endswith('.png'):
                raise ValueError(f'Invalid PNG filename: {fname!r}')
            png = folder / fname
            if not png.is_file():
                raise FileNotFoundError(f'PNG missing: {png}')
            if not row['frame_index'].isdigit():
                raise ValueError(f'Invalid frame_index: {j["name"]}/{sample}')
            eligible.append(row)
        if len(eligible) != len(whitelist[j['name']]) or len(eligible) < 15:
            raise ValueError(f'Whitelist join failed for {j["name"]}')
        prepared[j['name']] = {'job': j, 'folder': folder, 'rows': eligible}
    assert sum(len(v['rows']) for v in prepared.values()) == 3665
    return plan, jobs, prepared


def load_sources(root, plan, q):
    sources, masks = {}, []
    for video in ('shields', 'fourpeople'):
        path = root / plan['videos'][video]['path']
        if sha256_file(path) != plan['videos'][video]['sha256']:
            raise RuntimeError(f'Source Y4M SHA256 mismatch: {path}')
        meta_path = path.with_name(path.stem + '_numbered_metadata.json')
        if not meta_path.is_file():
            raise FileNotFoundError(f'Numbered-video metadata missing: {meta_path}')
        meta = read_json(meta_path)
        w, h, fps, offsets = q.index_y4m(path)
        if (w, h, len(offsets)) != (1280, 720, int(meta['frame_count'])):
            raise ValueError(f'Source geometry/frame count mismatch: {video}')
        masks.append(q.valid_pixel_mask((360, 640, 3), meta)[0])
        sources[video] = {'path': path, 'w': w, 'h': h, 'offsets': offsets}
    mask = np.logical_and.reduce(masks)
    if mask.mean() < .5:
        raise ValueError('Common mask covers too much of the image')
    return sources, mask


def model_fingerprint(model):
    """Full SHA256 over model weights, including parameter names and shapes."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode('utf-8'))
        h.update(str(tuple(tensor.shape)).encode('ascii'))
        raw = tensor.detach().contiguous().cpu()
        if raw.dtype == __import__('torch').bfloat16:
            raw = raw.view(__import__('torch').uint16)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


def load_models(device):
    import torch
    import torch.nn.functional as F
    from DISTS_pytorch import DISTS
    import open_clip

    print('[MODELS] Loading DISTS...', flush=True)
    dists = DISTS().eval().to(device)
    print('[MODELS] Loading DINOv2-Base (first run may download weights)...', flush=True)
    dino = torch.hub.load(DINO_REPO, DINO_NAME, trust_repo=True).eval().to(device)
    print('[MODELS] Loading OpenCLIP (first run may download weights)...', flush=True)
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_NAME, pretrained=OPENCLIP_WEIGHTS, device=device)
    clip.eval()
    fingerprints = {
        'dists_weights_sha256': model_fingerprint(dists),
        'dinov2_base_weights_sha256': model_fingerprint(dino),
        'openclip_weights_sha256': model_fingerprint(clip),
    }
    print('[MODELS] Model weights fingerprinted.', flush=True)
    return dists, dino, clip, clip_preprocess, fingerprints, F


def prepare_pair(reference_bgr, received_bgr, mask, clip_preprocess, torch):
    # Exclude barcode/overlay in BOTH images, without inserting ground truth in scoring.
    a = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(received_bgr, cv2.COLOR_BGR2RGB)
    a = a.copy()
    b = b.copy()
    a[~mask] = NEUTRAL_RGB
    b[~mask] = NEUTRAL_RGB
    a_pil, b_pil = Image.fromarray(a), Image.fromarray(b)
    a_dists = torch.from_numpy(a.transpose(2, 0, 1).copy()).float().div_(255.0)
    b_dists = torch.from_numpy(b.transpose(2, 0, 1).copy()).float().div_(255.0)
    a_small = ImageOps.pad(a_pil, DEEP_SIZE, method=Image.Resampling.BICUBIC,
                           color=NEUTRAL_RGB, centering=(.5, .5))
    b_small = ImageOps.pad(b_pil, DEEP_SIZE, method=Image.Resampling.BICUBIC,
                           color=NEUTRAL_RGB, centering=(.5, .5))
    dino_mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(3, 1, 1)
    dino_std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(3, 1, 1)
    a_dino = (torch.from_numpy(np.array(a_small).transpose(2, 0, 1).copy()).float() / 255 - dino_mean) / dino_std
    b_dino = (torch.from_numpy(np.array(b_small).transpose(2, 0, 1).copy()).float() / 255 - dino_mean) / dino_std
    return (a_dists, b_dists, a_dino, b_dino,
            clip_preprocess(a_small), clip_preprocess(b_small))


def get_embeddings(model, images, keys, cache, device, fn, F):
    missing = []
    seen = set()
    for key, im in zip(keys, images):
        if key not in cache and key not in seen:
            missing.append((key, im))
            seen.add(key)
    if missing:
        with __import__('torch').inference_mode():
            features = fn(model, __import__('torch').stack([x for _, x in missing]).to(device))
            if isinstance(features, dict):
                raise TypeError('Unexpected model output dict; expected embedding tensor')
            features = F.normalize(features.float(), dim=-1).cpu()
        for (key, _), vector in zip(missing, features):
            cache[key] = vector
    return __import__('torch').stack([cache[k] for k in keys]).to(device)


def process_job(item, sources, mask, models, device, batch_size, ref_cache):
    import torch
    dists, dino, clip, clip_preprocess, _, F = models
    j, folder, rows = item['job'], item['folder'], item['rows']
    src = sources[j['video']]
    result_rows = []
    with src['path'].open('rb') as source:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            pairs, keys = [], []
            for r in chunk:
                idx = int(r['frame_index'])
                if not 0 <= idx < len(src['offsets']):
                    raise ValueError(f'Frame index out of range: {j["name"]}/{idx}')
                received = cv2.imread(str(folder / r['capture_filename']), cv2.IMREAD_COLOR)
                rh, rw = int(float(r['capture_height'])), int(float(r['capture_width']))
                if received is None or received.shape[:2] != (rh, rw):
                    raise RuntimeError(f'PNG missing or dimensions changed: {folder / r["capture_filename"]}')
                if (rw, rh) != TARGET:
                    received = cv2.resize(received, TARGET, interpolation=cv2.INTER_AREA)
                reference = __import__('exp04_quality_offline').read_reference(
                    source, src['offsets'][idx], src['w'], src['h'], *TARGET)
                pairs.append(prepare_pair(reference, received, mask, clip_preprocess, torch))
                keys.append((j['video'], idx))
            with torch.inference_mode():
                a = torch.stack([p[0] for p in pairs]).to(device)
                b = torch.stack([p[1] for p in pairs]).to(device)
                distances = dists(a, b, require_grad=False, batch_average=False)
                distances = distances.reshape(-1).float().cpu().numpy()
                if len(distances) != len(chunk):
                    raise RuntimeError('DISTS returned wrong batch length')
                dino_ref = get_embeddings(dino, [p[2] for p in pairs], keys,
                                          ref_cache['dino'], device, lambda m, x: m(x), F)
                dino_rec = F.normalize(dino(torch.stack([p[3] for p in pairs]).to(device)).float(), dim=-1)
                dino_cos = (dino_ref * dino_rec).sum(dim=-1).cpu().numpy()
                clip_ref = get_embeddings(clip, [p[4] for p in pairs], keys,
                                          ref_cache['clip'], device, lambda m, x: m.encode_image(x), F)
                clip_rec = F.normalize(clip.encode_image(torch.stack([p[5] for p in pairs]).to(device)).float(), dim=-1)
                clip_cos = (clip_ref * clip_rec).sum(dim=-1).cpu().numpy()
            for k, r in enumerate(chunk):
                d, v, c = float(distances[k]), float(dino_cos[k]), float(clip_cos[k])
                if not all(map(math.isfinite, [d, v, c])):
                    raise ValueError(f'Non-finite model result: {j["name"]}/{r["sample"]}')
                result_rows.append({
                    'job': j['name'], 'video': j['video'], 'block': j['block'],
                    'condition': j['condition'], 'severity': j['severity'],
                    'sample': int(r['sample']), 'fault_active': int(r['fault_active']),
                    'frame_index': int(r['frame_index']),
                    'native_width': int(float(r['capture_width'])),
                    'native_height': int(float(r['capture_height'])),
                    'dists_distance': f'{d:.9f}', 'dists_similarity': f'{1-d:.9f}',
                    'dinov2_base_cosine': f'{v:.9f}', 'openclip_cosine': f'{c:.9f}',
                })
    if len(result_rows) != len(rows):
        raise RuntimeError('Output sample count mismatch')
    return result_rows


def summarize_session(rows):
    r = rows[0]
    get = lambda col: np.array([float(x[col]) for x in rows], dtype=np.float64)
    d, ds, di, cl = (get(k) for k in ('dists_distance', 'dists_similarity',
                                      'dinov2_base_cosine', 'openclip_cosine'))
    return dict(job=r['job'], video=r['video'], block=r['block'],
                condition=r['condition'], severity=r['severity'], n=len(rows),
                n_fault_active=sum(int(x['fault_active']) for x in rows),
                mean_dists_distance=float(d.mean()), q90_dists_distance=float(np.quantile(d, .9)),
                mean_dists_similarity=float(ds.mean()), q10_dists_similarity=float(np.quantile(ds, .1)),
                mean_dinov2_base_cosine=float(di.mean()), q10_dinov2_base_cosine=float(np.quantile(di, .1)),
                mean_openclip_cosine=float(cl.mean()), q10_openclip_cosine=float(np.quantile(cl, .1)))


def load_cached_job(path, item):
    rows = read_csv(path)
    if set(rows[0]) != set(SAMPLE_COLS) or len(rows) != len(item['rows']):
        raise ValueError(f'Cached output invalid: {path}')
    expected = {int(x['sample']): x for x in item['rows']}
    if len({int(r['sample']) for r in rows}) != len(rows):
        raise ValueError(f'Duplicate cached samples: {path}')
    for r in rows:
        key = int(r['sample'])
        if (key not in expected or r['job'] != item['job']['name'] or
                int(r['frame_index']) != int(expected[key]['frame_index']) or
                int(r['fault_active']) != int(expected[key]['fault_active'])):
            raise ValueError(f'Cached row inconsistent with input: {path} / {key}')
        for metric in ('dists_distance', 'dinov2_base_cosine', 'openclip_cosine'):
            if not math.isfinite(float(r[metric])):
                raise ValueError(f'Invalid cached score: {path}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=r'D:\visual-oracle-revision')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs; no models loaded')
    parser.add_argument('--max-new', type=int, default=None, help='Process only N new sessions, then stop')
    parser.add_argument('--batch-size', type=int, default=2, help='Lower to 1 if CUDA out of memory')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 8 or (args.max_new is not None and args.max_new < 1):
        parser.error('Batch size must be 1..8 and --max-new must be >= 1')
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    print('[PREFLIGHT] Validating plan, sample selection, 120 job statuses and PNG paths...', flush=True)
    plan, jobs, prepared = load_calibration(root)
    q = import_quality(root)
    # Register module for consistent reference-frame decoding in process_job.
    sys.modules['exp04_quality_offline'] = q
    sources, mask = load_sources(root, plan, q)
    print(f'[PREFLIGHT] PASS: {len(jobs)} sessions; '
          f'{sum(len(x["rows"]) for x in prepared.values())} PNGs; '
          f'common ROI {mask.mean()*100:.2f}%; plan SHA256 {PLAN_SHA256}', flush=True)
    if args.check_only:
        print('[PREFLIGHT] No model loaded and no experiment file modified.', flush=True)
        return

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but not available in this Python environment')
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cpu':
        print('[WARNING] CPU mode: 3665 image pairs may take many hours.', flush=True)
    print(f'[DEVICE] {device} | PyTorch {torch.__version__}', flush=True)
    torch.manual_seed(20261106)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261106)
    torch.backends.cudnn.benchmark = False

    out = root / 'captures' / 'exp04_deep_calibration'
    jobs_dir = out / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    mask_hash = sha256_bytes(np.ascontiguousarray(mask).tobytes())
    protocol = {
        'protocol': 'EXP04_deep_calibration_v1',
        'plan_sha256': PLAN_SHA256, 'selected_640_samples_sha256': SAMPLES_SHA256,
        'resolution': [640, 360], 'no_upscaling': True,
        'common_mask_sha256': mask_hash, 'mask_eligible_fraction': round(float(mask.mean()), 9),
        'mask_policy': 'excluded pixels replaced with RGB (127,127,127) in both images',
        'deep_preprocess': 'entire 640x360 frame letterboxed to 224x224; no center crop',
        'dists': 'DISTS_pytorch pretrained, RGB [0,1], distance',
        'dinov2': DINO_REPO + ':' + DINO_NAME + ', CLS cosine',
        'openclip': OPENCLIP_NAME + ':' + OPENCLIP_WEIGHTS + ', image cosine',
        'session_statistics': 'q90 DISTS distance, q10 DINO/OpenCLIP cosine; all eligible frames',
        'python': platform.python_version(), 'torch': torch.__version__,
        'opencv': cv2.__version__,
        'holdout_used': False, 'thresholds_selected': False,
    }
    models = load_models(device)
    protocol.update(models[4])
    proto_path = out / 'protocol.json'
    if proto_path.exists():
        old = read_json(proto_path)
        if old != protocol:
            raise RuntimeError('Model/protocol changed since checkpoint. Refusing to mix scores. '
                               f'Inspect existing directory: {out}')
    else:
        atomic_json(proto_path, protocol)

    cached = {}
    for job in jobs:
        path = jobs_dir / (job['name'] + '.csv')
        if path.exists():
            cached[job['name']] = load_cached_job(path, prepared[job['name']])
    print(f'[RESUME] {len(cached)}/120 complete deep-scoring session files.', flush=True)
    new_count = 0
    ref_cache = {'dino': {}, 'clip': {}}
    for idx, job in enumerate(jobs, 1):
        if job['name'] in cached:
            continue
        started = time.monotonic()
        print(f'[{idx:03d}/120] {job["name"]} ({len(prepared[job["name"]]["rows"])} frames)', flush=True)
        rows = process_job(prepared[job['name']], sources, mask, models, device,
                           args.batch_size, ref_cache)
        path = jobs_dir / (job['name'] + '.csv')
        write_csv_atomic(path, SAMPLE_COLS, rows)
        cached[job['name']] = rows
        new_count += 1
        print(f'    saved {len(rows)} rows in {time.monotonic()-started:.1f}s; '
              f'{len(cached)}/120 completed', flush=True)
        if args.max_new is not None and new_count >= args.max_new:
            break

    if len(cached) != 120:
        print(f'[INCOMPLETE] {len(cached)}/120; rerun the SAME command without --max-new to resume.', flush=True)
        return
    samples = [r for job in jobs for r in cached[job['name']]]
    sessions = [summarize_session(cached[job['name']]) for job in jobs]
    if len(samples) != 3665 or len(sessions) != 120:
        raise RuntimeError('Final aggregation count mismatch')
    sample_out = out / 'EXP04_deep_calibration_samples.csv'
    session_out = out / 'EXP04_deep_calibration_sessions.csv'
    write_csv_atomic(sample_out, SAMPLE_COLS, samples)
    write_csv_atomic(session_out, SESSION_COLS, sessions)
    audit = dict(protocol=protocol, sessions=len(sessions), frames=len(samples),
                 by_video=dict(Counter(r['video'] for r in sessions)),
                 by_condition=dict(Counter(r['condition'] for r in sessions)),
                 input_plan_sha256=PLAN_SHA256,
                 deep_samples_sha256=sha256_file(sample_out),
                 deep_sessions_sha256=sha256_file(session_out),
                 completed_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 no_thresholds_selected=True, holdout_used=False)
    audit_path = out / 'EXP04_deep_calibration_audit.json'
    atomic_json(audit_path, audit)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zpath = root / 'captures' / f'EXP04_deep_calibration_{stamp}.zip'
    with ZipFile(zpath, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in (sample_out, session_out, audit_path, proto_path):
            z.write(p, p.name)
    print(f'\n[COMPLETE] 120 sessions, 3665 frames. UPLOAD: {zpath}', flush=True)
    print('Holdout remains gated. No calibration lock has been written.', flush=True)


if __name__ == '__main__':
    main()
