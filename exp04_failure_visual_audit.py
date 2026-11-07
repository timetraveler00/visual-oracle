#!/usr/bin/env python3
"""EXP04 post-hoc, read-only visual failure audit (NO Jitsi, NO model fitting).

Reproduces paired clean/sender/received images for all 10 Parkrun baseline
false positives and 5 ScreenShare low-blur false negatives. Adds deterministic
ScreenShare baseline and medium-blur controls, source-motion measurements,
and immutable provenance. Never changes the frozen classifier or thresholds.

Windows Anaconda Prompt, D:\\visual-oracle-revision:
    python exp04_failure_visual_audit.py --check-only
    python exp04_failure_visual_audit.py

Input ZIP defaults to captures/EXP04_holdout_evaluation_20261107_071300.zip.
The result is a NEW captures/EXP04_failure_visual_audit_*.zip.
Exploratory diagnostics are NOT an independent holdout re-evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np

import exp04_score_holdout as scoring
import exp04_deep_calibration as deep

PKG_NAME = 'EXP04_holdout_evaluation_20261107_071300.zip'
SAMPLE_NAME = 'EXP04_holdout_5metric_samples.csv'
SESSION_NAME = 'EXP04_holdout_5metric_sessions.csv'
PRED_NAME = 'EXP04_holdout_frozen_predictions.csv'
METRIC_KEYS = ('ssim', 'psnr_db', 'dists_distance', 'dinov2_base_cosine', 'openclip_cosine')


def csv_from_zip(z, name):
    with z.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def validate_package(z):
    required = {SAMPLE_NAME, SESSION_NAME, PRED_NAME,
                'EXP04_holdout_frozen_evaluation.json',
                'EXP04_holdout_scoring_audit.json', 'protocol.json',
                'EXP04_holdout_checksums.json'}
    if set(z.namelist()) != required:
        raise RuntimeError('Unexpected evaluation ZIP contents; refusing to mix files')
    checks = json.loads(z.read('EXP04_holdout_checksums.json'))
    for filename in required - {'EXP04_holdout_checksums.json'}:
        actual = hashlib.sha256(z.read(filename)).hexdigest()
        if checks.get(filename) != actual:
            raise RuntimeError(f'Evaluation ZIP checksum mismatch: {filename}')
    audit = json.loads(z.read('EXP04_holdout_scoring_audit.json'))
    if (audit.get('stage') != 'holdout' or audit.get('completed_jobs') != 120 or
        audit.get('eligible_frames') != 3538 or
        audit.get('plan_sha256') != scoring.PLAN_SHA256 or
        audit.get('calibration_lock_sha256') != scoring.LOCK_SHA256 or
        audit.get('frozen_scorer_sha256') != scoring.FROZEN_SHA256):
        raise RuntimeError('Unexpected experiment audit/lock metadata')
    samples = csv_from_zip(z, SAMPLE_NAME)
    sessions = csv_from_zip(z, SESSION_NAME)
    preds = csv_from_zip(z, PRED_NAME)
    if len(samples) != 3538 or len(sessions) != 120 or len(preds) != 120:
        raise RuntimeError('Holdout CSV row counts unexpected')
    if len({r['job'] for r in preds}) != 120 or len({r['job'] for r in sessions}) != 120:
        raise RuntimeError('Duplicate or missing holdout job in CSV')
    if len({(r['job'], int(r['sample'])) for r in samples}) != len(samples):
        raise RuntimeError('Duplicate sample ID in evaluation ZIP')
    return samples, sessions, preds, audit


def identify_cases(samples, preds):
    by_job = defaultdict(list)
    for row in samples:
        by_job[row['job']].append(row)
    for rows in by_job.values():
        rows.sort(key=lambda x: int(x['sample']))
    fp = [p for p in preds if int(p['actual_fault']) == 0 and int(p['predict_LAYERED_PRIMARY']) == 1]
    fn = [p for p in preds if int(p['actual_fault']) == 1 and int(p['predict_LAYERED_PRIMARY']) == 0]
    if (len(fp), len(fn)) != (10, 5):
        raise RuntimeError(f'Expected 10 FP and 5 FN in frozen evaluation, got {len(fp)}, {len(fn)}')
    if any(p['video'] != 'parkrun' or p['condition'] != 'baseline' for p in fp):
        raise RuntimeError('Unexpected FP type')
    if any(p['video'] != 'screenshare' or p['condition'] != 'blur' or p['severity'] != 'low' for p in fn):
        raise RuntimeError('Unexpected FN type')
    cases = []
    for p in sorted(fp, key=lambda x: x['job']):
        cases.append((p, 'FP_parkrun_baseline'))
    for p in sorted(fn, key=lambda x: x['job']):
        cases.append((p, 'FN_screenshare_blur_low'))
    for block in range(1, 6):
        for condition, severity, role in (
            ('baseline', 'none', 'CONTROL_screenshare_baseline'),
            ('blur', 'medium', 'CONTROL_screenshare_blur_medium')):
            eligible = sorted((p for p in preds if p['video'] == 'screenshare' and
                               int(p['block']) == block and p['condition'] == condition and
                               p['severity'] == severity), key=lambda x: x['job'])
            if not eligible:
                raise RuntimeError(f'Missing control {block}/{condition}/{severity}')
            cases.append((eligible[0], role))
    assert len(cases) == 25
    for p, role in cases:
        if p['job'] not in by_job:
            raise RuntimeError(f'Missing sample scores: {p["job"]}')
    return cases, by_job


def selected_rows(rows, role):
    ordered = sorted(rows, key=lambda r: (float(r['ssim']), int(r['sample'])))
    if role.startswith('CONTROL'):
        return [ordered[len(ordered)//2]]
    targets = [ordered[0], ordered[len(ordered)//2]]
    if role.startswith('FN_'):
        targets.append(max(rows, key=lambda r: (float(r['dists_distance']), -int(r['sample']))))
    else:
        targets.append(ordered[-1])
    used = set()
    out = []
    for r in targets + list(reversed(ordered)):
        k = int(r['sample'])
        if k not in used:
            out.append(r)
            used.add(k)
        if len(out) == 3:
            break
    return out


def safe_png(img, path):
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f'Could not write image: {path}')


def panel(bgr, label, mask, w=480, h=270):
    masked = bgr.copy()
    masked[~mask] = 127
    small = cv2.resize(masked, (w, h), interpolation=cv2.INTER_AREA)
    header = np.full((43, w, 3), 28, dtype=np.uint8)
    cv2.putText(header, label[:50], (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((header, small), axis=0)


def diff_panel(ref, rec, mask):
    # Fixed 3x amplification, NOT per-image min-max normalization.
    delta = np.mean(cv2.absdiff(ref, rec), axis=2)
    amplified = np.uint8(np.clip(delta * 3.0, 0, 255))
    heat = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)
    return panel(heat, 'ABS RGB ERROR x3 (fixed)', mask)


def get_ref(f, offsets, idx, w, h, q):
    return q.read_reference(f, offsets[idx], w, h, 640, 360)


def mae_masked(a, b, mask):
    return float(np.mean(np.abs(a.astype(np.float32)-b.astype(np.float32))[mask]))


def motion_value(f, source, idx, q, mask):
    # 5 frames at 50 FPS = 100ms. Avoid wraparound at clip boundaries.
    neighbor = idx + 5 if idx <= len(source['offsets']) - 6 else idx - 5
    ref = get_ref(f, source['offsets'], idx, source['w'], source['h'], q)
    near = get_ref(f, source['offsets'], neighbor, source['w'], source['h'], q)
    return mae_masked(ref, near, mask)


def find_sender(root, item, source, cache, q):
    job = item['job']
    if job['condition'] == 'baseline':
        return None
    settings_path = item['folder'] / 'settings.json'
    settings = deep.read_json(settings_path)
    fault = settings.get('fault_injection') or {}
    expected = fault.get('source_sha256')
    basename = fault.get('source_video')
    if not expected or not basename or Path(basename).name != basename:
        return None
    key = (job['video'], job['condition'], job['severity'], expected)
    if key in cache:
        return cache[key]
    fault_root = root / 'fault_sources'
    candidates = list(fault_root.glob('*/' + basename))
    candidates = [p for p in candidates if p.parent.name.startswith(source['path'].stem + '_')]
    matches = [p for p in candidates if digest(p) == expected]
    if len(matches) != 1:
        print(f'[WARN] Sender fault Y4M not uniquely found for {job["name"]}: {len(matches)} match(es)')
        cache[key] = None
        return None
    path = matches[0]
    w, h, fps, offsets = q.index_y4m(path)
    if (w, h, len(offsets)) != (1280, 720, 500) or abs(fps-50) > 1e-6:
        raise RuntimeError(f'Unexpected sender Y4M: {path}')
    cache[key] = {'path': path, 'offsets': offsets, 'w': w, 'h': h, 'sha256': expected}
    return cache[key]


def run(root, evaluation, check_only=False):
    if not root.is_dir() or not evaluation.is_file():
        raise FileNotFoundError(f'Project root / evaluation ZIP missing: {root}, {evaluation}')
    print('[LOCK] Checking plan, calibration lock, and unchanged frozen scorer...', flush=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    plan, lock = scoring.check_lock(root)
    q = deep.import_quality(root)
    print('[PACKAGE] Checking evaluation ZIP checksums and 120 frozen predictions...', flush=True)
    with ZipFile(evaluation) as z:
        samples, sessions, preds, audit = validate_package(z)
    cases, by_job = identify_cases(samples, preds)
    print(f'[CASES] {sum(role.startswith("FP_") for _,role in cases)} false positives, '
          f'{sum(role.startswith("FN_") for _,role in cases)} false negatives, '
          f'10 controls; 25 contact sheets planned.', flush=True)
    print('[CAPTURES] Checking 120 original holdout runs and source Y4M hashes...', flush=True)
    jobs, sources, mask, mask_sha, prepared, exclusions = scoring.prepare_holdout(root, plan, lock, q)
    if mask_sha != audit['common_mask_sha256'] or sum(map(lambda x: len(x['rows']), prepared.values())) != len(samples):
        raise RuntimeError('Capture/ROI mismatch vs evaluation package')
    by_source = {j['name']: j for j in jobs}
    for job, rows in by_job.items():
        if job not in prepared:
            raise RuntimeError(f'Unknown evaluation job: {job}')
        expected = {int(r['sample']): r for r in prepared[job]['rows']}
        if set(int(r['sample']) for r in rows) != set(expected):
            raise RuntimeError(f'Sample set mismatch: {job}')
        for r in rows:
            e = expected[int(r['sample'])]
            if (int(r['frame_index']) != int(e['frame_index']) or
                int(r['fault_active']) != int(e['fault_active']) or
                int(r['native_width']) != int(float(e['capture_width'])) or
                int(r['native_height']) != int(float(e['capture_height'])) or
                r['video'] != by_source[job]['video'] or
                r['condition'] != by_source[job]['condition']):
                raise RuntimeError(f'Sample metadata mismatch: {job}/{r["sample"]}')
    print('[PASS] All 3538 sample records match original capture metadata.', flush=True)
    if check_only:
        print('[CHECK-ONLY] No files changed; run without --check-only to create visual audit.', flush=True)
        return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = root / 'captures' / ('EXP04_failure_visual_audit_' + stamp)
    out.mkdir(parents=False, exist_ok=False)
    images = out / 'contact_sheets'
    images.mkdir()
    sender_cache = {}
    selection = []
    for p, role in cases:
        job = p['job']
        rows = by_job[job]
        chosen = selected_rows(rows, role)
        item = prepared[job]
        src = sources[p['video']]
        sender = find_sender(root, item, src, sender_cache, q)
        display_rows = []
        with src['path'].open('rb') as clean_f:
            sender_f = sender['path'].open('rb') if sender else None
            try:
                by_sample = {int(r['sample']): r for r in item['rows']}
                for r in chosen:
                    idx = int(r['frame_index'])
                    sample = int(r['sample'])
                    input_row = by_sample[sample]
                    png = item['folder'] / input_row['capture_filename']
                    rec = cv2.imread(str(png), cv2.IMREAD_COLOR)
                    native = (int(r['native_width']), int(r['native_height']))
                    if rec is None or rec.shape[1::-1] != native:
                        raise RuntimeError(f'Capture PNG dimensions changed: {png}')
                    if native != (640, 360):
                        rec = cv2.resize(rec, (640, 360), interpolation=cv2.INTER_AREA)
                    ref = get_ref(clean_f, src['offsets'], idx, src['w'], src['h'], q)
                    ssim, psnr = q.quality_metrics(ref, rec, mask)
                    if abs(ssim-float(r['ssim'])) > 2e-5 or abs(psnr-float(r['psnr_db'])) > 2e-4:
                        raise RuntimeError(f'PNG/reference scores changed: {job}/{sample} '
                                           f'({ssim:.6f}/{psnr:.5f} vs {r["ssim"]}/{r["psnr_db"]})')
                    injected = None
                    injected_ssim = injected_psnr = rec_sender_ssim = rec_sender_psnr = ''
                    if sender_f:
                        injected = get_ref(sender_f, sender['offsets'], idx, sender['w'], sender['h'], q)
                        injected_ssim, injected_psnr = q.quality_metrics(ref, injected, mask)
                        rec_sender_ssim, rec_sender_psnr = q.quality_metrics(injected, rec, mask)
                    display = [panel(ref, f'CLEAN | frame {idx}', mask),
                               panel(injected if injected is not None else ref,
                                     'INJECTED SENDER' if injected is not None else 'SENDER NOT AVAILABLE / CLEAN', mask),
                               panel(rec, f'RECEIVED | sample {sample}', mask),
                               diff_panel(ref, rec, mask)]
                    display_rows.append(np.concatenate(display, axis=1))
                    selection.append({
                        'role': role, 'job': job, 'video': p['video'], 'block': p['block'],
                        'condition': p['condition'], 'severity': p['severity'],
                        'sample': sample, 'frame_index': idx, 'native_width': native[0],
                        'native_height': native[1], 'fault_active': r['fault_active'],
                        'ssim_clean_received': f'{ssim:.9f}', 'psnr_clean_received_db': f'{psnr:.9f}',
                        'dists_distance': r['dists_distance'], 'dinov2_base_cosine': r['dinov2_base_cosine'],
                        'openclip_cosine': r['openclip_cosine'],
                        'sender_available': int(sender is not None),
                        'ssim_clean_sender': f'{injected_ssim:.9f}' if injected is not None else '',
                        'psnr_clean_sender_db': f'{injected_psnr:.9f}' if injected is not None else '',
                        'ssim_sender_received': f'{rec_sender_ssim:.9f}' if injected is not None else '',
                        'psnr_sender_received_db': f'{rec_sender_psnr:.9f}' if injected is not None else '',
                        'capture_png_sha256': digest(png),
                        'sender_y4m_sha256': sender['sha256'] if sender else '',
                    })
            finally:
                if sender_f:
                    sender_f.close()
        sheet = np.concatenate(display_rows, axis=0)
        safe_png(sheet, images / (job + '.png'))
        print(f'[IMAGE] {role}: {job} ({len(chosen)} frames)', flush=True)

    # Motion proxy on ALL 593 eligible baseline frames (both videos), not just selected frames.
    motion_rows = []
    for video in ('parkrun', 'screenshare'):
        src = sources[video]
        with src['path'].open('rb') as f:
            for p in sorted((x for x in preds if x['video'] == video and x['condition'] == 'baseline'),
                            key=lambda x: x['job']):
                for r in by_job[p['job']]:
                    idx = int(r['frame_index'])
                    motion = motion_value(f, src, idx, q, mask)
                    motion_rows.append({'job': p['job'], 'video': video,
                                        'sample': r['sample'], 'frame_index': idx,
                                        'native_width': r['native_width'],
                                        'ssim': r['ssim'], 'psnr_db': r['psnr_db'],
                                        'source_motion_mae_100ms': f'{motion:.7f}'})
        d = [float(r['source_motion_mae_100ms']) for r in motion_rows if r['video'] == video]
        print(f'[MOTION] {video}: n={len(d)}, median 100ms source MAE={np.median(d):.3f}', flush=True)
    if len(motion_rows) != 593:
        raise RuntimeError(f'Unexpected baseline sample count: {len(motion_rows)} != 593')

    def write_csv(path, rows):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_csv(out / 'selected_frame_diagnostics.csv', selection)
    write_csv(out / 'all_baseline_source_motion.csv', motion_rows)
    case_rows = [{**{k: p[k] for k in ('job', 'video', 'block', 'condition', 'severity', 'n',
                                       'q10_ssim', 'q10_psnr_db', 'q90_dists_distance',
                                       'q10_dinov2_base_cosine', 'q10_openclip_cosine',
                                       'actual_fault', 'predict_LAYERED_PRIMARY',
                                       'layer_1_structural',
                                       'layer_2_perceptual_semantic_consistency',
                                       'layer_3_semantic_break')}, 'audit_role': role}
                 for p, role in cases]
    write_csv(out / 'case_manifest.csv', case_rows)
    summary = {
        'purpose': 'Post-hoc exploratory failure analysis, NOT threshold selection',
        'input_evaluation_zip': str(evaluation),
        'input_evaluation_sha256': digest(evaluation),
        'plan_sha256': scoring.PLAN_SHA256,
        'calibration_lock_sha256': scoring.LOCK_SHA256,
        'frozen_scorer_sha256': scoring.FROZEN_SHA256,
        'common_mask_sha256': mask_sha,
        'number_of_cases': 15,
        'number_of_controls': 10,
        'contact_sheets': len(cases),
        'selected_frames': len(selection),
        'baseline_motion_samples': len(motion_rows),
        'sender_fault_video_available_in_selected_frames': sum(int(r['sender_available']) for r in selection),
        'selection_rule': 'FP: min/median/max SSIM; FN: min/median SSIM + max DISTS (unique fallback); controls: median SSIM',
        'motion_proxy': 'Clean source 640x360 masked RGB MAE between frame index and +/-5 frames (100ms at 50fps)',
        'image_policy': 'Common calibration mask applied to all panels; abs RGB error heatmap fixed 3x gain',
        'note': 'Only new output folder is written; original images, lock, frozen scorer, thresholds untouched.',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    (out / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'README.txt').write_text(
        'EXP04 VISUAL FAILURE AUDIT\n'
        'Exploratory only; does not revise locked holdout evaluation.\n'
        '25 contact sheets: 10 Parkrun baseline FP, 5 ScreenShare low-blur FN,\n'
        '5 ScreenShare baseline controls, 5 ScreenShare medium-blur controls.\n'
        'Panels: CLEAN | INJECTED SENDER (if source available) | RECEIVED | ABS RGB ERROR x3.\n'
        'All panels exclude the original frozen calibration ROI mask.\n'
        'Motion CSV compares clean-source frames 100ms apart, not a direct network measurement.\n'
        'No new Jitsi meetings; no re-scoring or threshold optimization.\n', encoding='utf-8')
    checks = {}
    for p in sorted(out.rglob('*')):
        if p.is_file():
            checks[p.relative_to(out).as_posix()] = digest(p)
    (out / 'file_checksums.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    archive = out.with_suffix('.zip')
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    print(f'\n[COMPLETE] Exploratory visual audit: {archive}', flush=True)
    print('[UNCHANGED] 240 Jitsi runs, original captures, calibration lock and frozen scorer.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=r'D:\visual-oracle-revision')
    parser.add_argument('--evaluation', default=None, help='Full path to EXP04_holdout_evaluation_*.zip')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    evaluation = Path(args.evaluation).resolve() if args.evaluation else root / 'captures' / PKG_NAME
    run(root, evaluation, args.check_only)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
