#!/usr/bin/env python3
"""EXP04 exploratory post-hoc SPATIAL registration / blur-like attenuation audit.

For all 593 eligible Parkrun/ScreenShare holdout BASELINE frames:
  1. Reproduce the locked SSIM and PSNR at the barcode source frame.
  2. Estimate residual source-vs-receiver XY displacement by phase correlation.
  3. Apply an *exploratory* inverse shift (if reliable and >=0.25 px), comparing
     before/after on an identical eroded ROI mask.
  4. Measure paired gradient and high-pass energy ratios (receiver/source).
  5. Compare received images with a small family of Gaussian-smoothed *clean*
     references. The best match is descriptive, not a valid new oracle.

This is NOT a classifier or a recalibration. It must not change the frozen
oracle, thresholds, holdout scores, Jitsi sessions, or any source/capture files.

Requires pre-existing EXP04 scripts (already installed in project):
  exp04_temporal_alignment_audit.py
  exp04_score_holdout.py
  exp04_deep_calibration.py

Windows Anaconda Prompt:
  python exp04_spatial_blur_audit.py --check-only
  python exp04_spatial_blur_audit.py --smoke 2
  python exp04_spatial_blur_audit.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

import exp04_score_holdout as scoring
import exp04_deep_calibration as deep
import exp04_temporal_alignment_audit as temporal

SIGMAS = (0.0, 0.6, 1.0, 1.5, 2.0, 3.0)
EXPECTED = {'parkrun': 289, 'screenshare': 304}
INNER_MARGIN = 8
MIN_PHASE_RESPONSE = 0.15
MIN_CORRECTABLE_SHIFT = 0.25
MAX_CORRECTABLE_SHIFT = 3.0
TARGET = (640, 360)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def phase_displacement(reference: np.ndarray, received: np.ndarray, mask: np.ndarray):
    """Estimate receiver translation relative to reference on an overlay-free crop.

    The central crop avoids the upper-left barcode region and masked boundary.
    Gaussian smoothing suppresses codec noise; Hanning reduces boundary effects.
    The phase response is diagnostic, not a calibrated confidence probability.
    """
    x0, x1, y0, y1 = 40, 600, 95, 340
    if not mask[y0:y1, x0:x1].all():
        raise RuntimeError('Registration crop intersects the locked excluded ROI')
    a = cv2.cvtColor(reference[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(received[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = cv2.GaussianBlur(a, (5, 5), 1.0)
    b = cv2.GaussianBlur(b, (5, 5), 1.0)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    if not all(math.isfinite(v) for v in (dx, dy, response)):
        raise RuntimeError('Non-finite phase-correlation result')
    return float(dx), float(dy), float(response)


def interior_mask(mask: np.ndarray):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8),
                      iterations=1).astype(bool)
    inner[:INNER_MARGIN, :] = False
    inner[-INNER_MARGIN:, :] = False
    inner[:, :INNER_MARGIN] = False
    inner[:, -INNER_MARGIN:] = False
    if np.mean(inner) < 0.70:
        raise RuntimeError('Insufficient interior ROI')
    return inner


def corrected_image(received: np.ndarray, dx: float, dy: float):
    # phaseCorrelate(a,b) estimates displacement of b relative to a.
    affine = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(received, affine, TARGET, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def detail_ratios(reference: np.ndarray, received: np.ndarray, mask: np.ndarray):
    """Paired texture proxies, NOT codec-identification or causal measures."""
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(received, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def gradient_energy(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(cv2.magnitude(gx, gy)[mask]))

    def highpass_energy(x):
        hp = x - cv2.GaussianBlur(x, (0, 0), sigmaX=1.2, sigmaY=1.2)
        return float(np.mean(np.abs(hp[mask])))

    ga, gb = gradient_energy(a), gradient_energy(b)
    ha, hb = highpass_energy(a), highpass_energy(b)
    return (gb / ga if ga > 1e-6 else float('nan'),
            hb / ha if ha > 1e-6 else float('nan'),
            ga, gb, ha, hb)


def blur_family(reference: np.ndarray, received: np.ndarray, mask: np.ndarray, q):
    rows = []
    for sigma in SIGMAS:
        candidate = reference if sigma == 0 else cv2.GaussianBlur(
            reference, (0, 0), sigmaX=sigma, sigmaY=sigma)
        ssim, psnr = q.quality_metrics(candidate, received, mask)
        if not math.isfinite(ssim) or math.isnan(psnr):
            raise RuntimeError('Invalid blur-family score')
        rows.append({'sigma': sigma, 'ssim': ssim, 'psnr_db': psnr})
    # Prefer the unblurred source in case of ties.
    best = max(rows, key=lambda r: (r['ssim'], -r['sigma']))
    return rows, best


def inspect_pair(reference, received, mask, inner, q, frozen_ssim, frozen_psnr):
    s0, p0 = q.quality_metrics(reference, received, mask)
    if (abs(s0 - frozen_ssim) > temporal.SSIM_TOL or
            abs(p0 - frozen_psnr) > temporal.PSNR_TOL):
        raise RuntimeError(f'Frozen score mismatch: {s0:.9f}/{p0:.9f} '
                           f'!= {frozen_ssim:.9f}/{frozen_psnr:.9f}')
    dx, dy, response = phase_displacement(reference, received, mask)
    mag = math.hypot(dx, dy)
    apply_shift = bool(response >= MIN_PHASE_RESPONSE and
                       MIN_CORRECTABLE_SHIFT <= mag <= MAX_CORRECTABLE_SHIFT and
                       max(abs(dx), abs(dy)) <= MAX_CORRECTABLE_SHIFT)
    s_base, p_base = q.quality_metrics(reference, received, inner)
    rec_corr = corrected_image(received, dx, dy) if apply_shift else received
    s_corr, p_corr = q.quality_metrics(reference, rec_corr, inner)
    gradient_ratio, highpass_ratio, ga, gb, ha, hb = detail_ratios(
        reference, received, inner)
    blur_rows, best_blur = blur_family(reference, received, mask, q)
    return {
        'original_ssim': s0, 'original_psnr_db': p0,
        'phase_dx_px': dx, 'phase_dy_px': dy, 'phase_response': response,
        'phase_shift_magnitude_px': mag, 'phase_correction_applied': int(apply_shift),
        'spatial_base_ssim': s_base, 'spatial_corrected_ssim': s_corr,
        'spatial_ssim_gain': s_corr - s_base,
        'spatial_base_psnr_db': p_base, 'spatial_corrected_psnr_db': p_corr,
        'spatial_psnr_gain_db': p_corr - p_base,
        'gradient_ratio': gradient_ratio, 'highpass_ratio': highpass_ratio,
        'source_gradient_energy': ga, 'received_gradient_energy': gb,
        'source_highpass_energy': ha, 'received_highpass_energy': hb,
        'best_blur_sigma': best_blur['sigma'],
        'best_blur_ssim': best_blur['ssim'],
        'best_blur_ssim_gain': best_blur['ssim'] - s0,
        'best_blur_psnr_db': best_blur['psnr_db'],
        'best_blur_psnr_gain_db': best_blur['psnr_db'] - p0,
        'blur_rows': blur_rows,
    }


def median(rows, key):
    vals = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
    return float(np.median(vals)) if vals else float('nan')


def summarize(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for group, rr in sorted(groups.items()):
        s = {k: v for k, v in zip(keys, group)}
        s.update({
            'n': len(rr),
            'median_original_ssim': round(median(rr, 'original_ssim'), 7),
            'median_original_psnr_db': round(median(rr, 'original_psnr_db'), 5),
            'median_phase_shift_magnitude_px': round(median(rr, 'phase_shift_magnitude_px'), 5),
            'p90_phase_shift_magnitude_px': round(float(np.quantile(
                [float(r['phase_shift_magnitude_px']) for r in rr], .9)), 5),
            'phase_correction_applied_n': sum(int(r['phase_correction_applied']) for r in rr),
            'median_spatial_ssim_gain': round(median(rr, 'spatial_ssim_gain'), 7),
            'spatial_ssim_gain_ge_0p01_n': sum(float(r['spatial_ssim_gain']) >= .01 for r in rr),
            'spatial_ssim_gain_ge_0p03_n': sum(float(r['spatial_ssim_gain']) >= .03 for r in rr),
            'median_gradient_ratio': round(median(rr, 'gradient_ratio'), 6),
            'median_highpass_ratio': round(median(rr, 'highpass_ratio'), 6),
            'median_best_blur_sigma': round(median(rr, 'best_blur_sigma'), 3),
            'best_blur_sigma_positive_n': sum(float(r['best_blur_sigma']) > 0 for r in rr),
            'median_best_blur_ssim_gain': round(median(rr, 'best_blur_ssim_gain'), 7),
            'p90_best_blur_ssim_gain': round(float(np.quantile(
                [float(r['best_blur_ssim_gain']) for r in rr], .9)), 7),
            'best_blur_ssim_gain_ge_0p01_n': sum(float(r['best_blur_ssim_gain']) >= .01 for r in rr),
            'best_blur_ssim_gain_ge_0p03_n': sum(float(r['best_blur_ssim_gain']) >= .03 for r in rr),
        })
        out.append(s)
    return out


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f'No rows: {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(path: Path, reference, received, mask, result, idx):
    best_sigma = result['best_blur_sigma']
    smooth = reference if best_sigma == 0 else cv2.GaussianBlur(
        reference, (0, 0), sigmaX=best_sigma, sigmaY=best_sigma)
    images = [
        temporal.panel(reference, f'BARCODE SOURCE #{idx}', mask),
        temporal.panel(received, 'RECEIVED BASELINE', mask),
        temporal.panel(smooth, f'SOURCE GAUSS sigma={best_sigma:.1f}', mask),
        temporal.error_panel(reference, received, f'ORIGINAL SSIM={result["original_ssim"]:.3f}', mask),
        temporal.error_panel(smooth, received, f'SMOOTHED SSIM={result["best_blur_ssim"]:.3f}', mask),
    ]
    if not cv2.imwrite(str(path), np.concatenate(images, axis=1)):
        raise RuntimeError(f'Could not save contact sheet: {path}')


def findings_text(videos, segments):
    lines = [
        '# EXP04 spatial-registration and blur-like attenuation diagnostic', '',
        '**Exploratory post-hoc baseline-only analysis. Not a new oracle.**', '',
        '## By video', '',
        '| Video | Frames | Median XY shift (px) | Median gradient ratio | '
        'Median high-pass ratio | Median best Gaussian sigma | Median SSIM gain after smoothing source |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in videos:
        lines.append(f'| {r["video"]} | {r["n"]} | '
                     f'{r["median_phase_shift_magnitude_px"]:.3f} | '
                     f'{r["median_gradient_ratio"]:.3f} | '
                     f'{r["median_highpass_ratio"]:.3f} | '
                     f'{r["median_best_blur_sigma"]:.2f} | '
                     f'{r["median_best_blur_ssim_gain"]:+.4f} |')
    lines.extend(['', '## By source time segment', '',
                  '| Video | Segment | Frames | Median gradient ratio | Median blur SSIM gain |',
                  '|---|---|---:|---:|---:|'])
    for r in segments:
        lines.append(f'| {r["video"]} | {r["segment"]} | {r["n"]} | '
                     f'{r["median_gradient_ratio"]:.3f} | '
                     f'{r["median_best_blur_ssim_gain"]:+.4f} |')
    lines.extend(['', '## Important interpretation boundaries', '',
                  '- The barcode-matched original SSIM and PSNR are reproduced before any diagnostic.',
                  '- Phase-correlation displacement is a descriptive registration estimate, not proof of lag.',
                  '- XY correction uses a smaller but IDENTICAL interior mask before and after correction.',
                  '- Gradient/high-pass ratios are image-detail proxies, not codec measurements.',
                  '- Choosing the best Gaussian blur is an optimistic post-hoc match; it can also absorb '
                  'temporal averaging, resizing, and other degradation.',
                  '- A blur-like result does not identify the WebRTC codec, bitrate, or causal mechanism.',
                  '- No thresholds, oracle predictions, source videos, PNGs, or Jitsi runs are changed.',
                  '- Only two held-out source videos; image samples are not independent content types.', ''])
    return '\n'.join(lines)


def run(root: Path, evaluation: Path, check_only=False, smoke=None):
    if not root.is_dir() or not evaluation.is_file():
        raise FileNotFoundError(f'Project/evaluation ZIP missing: {root} / {evaluation}')
    os.chdir(root)
    sys.path.insert(0, str(root))
    print('[LOCK] Verifying original plan, active lock, and immutable scorer...', flush=True)
    plan, lock = scoring.check_lock(root)
    q = deep.import_quality(root)
    baseline, preds, audit = temporal.load_verified_evaluation(evaluation)
    print('[PREFLIGHT] Verifying 120 holdout COMPLETE jobs, source hashes, frozen ROI...', flush=True)
    jobs, sources, mask, mask_sha, prepared, exclusions = scoring.prepare_holdout(root, plan, lock, q)
    if mask_sha != audit['common_mask_sha256']:
        raise RuntimeError('Frozen ROI mask hash mismatch')
    inner = interior_mask(mask)
    by_job = defaultdict(list)
    for r in baseline:
        by_job[r['job']].append(r)
        if r['job'] not in prepared or r['video'] not in sources:
            raise RuntimeError(f'Unexpected baseline job: {r["job"]}')
    for job, rows in by_job.items():
        item = prepared[job]
        if item['job']['condition'] != 'baseline':
            raise RuntimeError(f'Not a baseline job: {job}')
        by_sample = {int(r['sample']): r for r in item['rows']}
        if set(int(r['sample']) for r in rows) != set(by_sample):
            raise RuntimeError(f'Frozen CSV vs capture sample mismatch: {job}')
        for r in rows:
            raw = by_sample[int(r['sample'])]
            if (int(r['frame_index']) != int(raw['frame_index']) or
                int(r['native_width']) != int(float(raw['capture_width'])) or
                int(r['native_height']) != int(float(raw['capture_height'])) or
                int(r['fault_active']) != 0):
                raise RuntimeError(f'Frozen capture mismatch: {job}/{r["sample"]}')
    if len(by_job) != 20 or Counter(r['video'] for r in baseline) != EXPECTED:
        raise RuntimeError('Unexpected baseline job/sample counts')
    print('[PASS] 20 baseline sessions / 593 frames. Original metric equality enforced.', flush=True)
    if check_only:
        print('[CHECK-ONLY] No images processed or files written.', flush=True)
        return

    rows_out = []
    worst_images = {}
    for video in ('parkrun', 'screenshare'):
        src = sources[video]
        meta_path = src['path'].with_name(src['path'].stem + '_numbered_metadata.json')
        meta = deep.read_json(meta_path)
        if (int(meta['width']), int(meta['height']), int(meta['frame_count'])) != (1280, 720, 500):
            raise RuntimeError(f'Unexpected barcode metadata: {meta_path}')
        jobs_this_video = sorted(j for j in by_job if prepared[j]['job']['video'] == video)
        processed = 0
        print(f'[VIDEO] {video}: {len(jobs_this_video)} baseline sessions', flush=True)
        with temporal.SourceCache(src, q) as cache:
            for job in jobs_this_video:
                item = prepared[job]
                originals = {int(r['sample']): r for r in item['rows']}
                rr = sorted(by_job[job], key=lambda r: int(r['sample']))
                worst_sample = int(min(rr, key=lambda r: (float(r['ssim']), int(r['sample'])))['sample'])
                for r in rr:
                    if smoke is not None and processed >= smoke:
                        break
                    sample = int(r['sample'])
                    raw = originals[sample]
                    png = item['folder'] / raw['capture_filename']
                    received = cv2.imread(str(png), cv2.IMREAD_COLOR)
                    native = (int(r['native_width']), int(r['native_height']))
                    if received is None or received.shape[1::-1] != native:
                        raise RuntimeError(f'PNG missing or changed: {png}')
                    barcode_idx, _, _ = q.decode_barcode(received, meta)
                    idx = int(r['frame_index'])
                    if barcode_idx != idx:
                        raise RuntimeError(f'Barcode mismatch: {job}/{sample}')
                    if native != TARGET:
                        received = cv2.resize(received, TARGET, interpolation=cv2.INTER_AREA)
                    ref = cache.frame(idx)
                    result = inspect_pair(ref, received, mask, inner, q,
                                          float(r['ssim']), float(r['psnr_db']))
                    if not all((not math.isnan(float(v))) and
                               (math.isfinite(float(v)) or 'psnr' in k)
                               for k, v in result.items() if k not in {'blur_rows'}):
                        raise RuntimeError(f'Non-finite metric: {job}/{sample}')
                    flat = {k: v for k, v in result.items() if k != 'blur_rows'}
                    row = {'job': job, 'video': video, 'block': r['block'],
                           'sample': sample, 'frame_index': idx,
                           'source_time_seconds': round(idx/50, 4),
                           'segment': '0-8s' if idx < 400 else '8-10s',
                           **{k: round(float(v), 9) for k, v in flat.items()}}
                    rows_out.append(row)
                    if smoke is None and sample == worst_sample:
                        worst_images[job] = (ref.copy(), received.copy(), result, idx)
                    processed += 1
                if smoke is not None and processed >= smoke:
                    break
                print(f'  [SESSION] {job}: {len(rr)} frames', flush=True)
        if smoke is not None:
            print(f'[SMOKE] {video}: {processed} frames, frozen scores reproduced.', flush=True)
    if smoke is not None:
        print(f'[SMOKE COMPLETE] {len(rows_out)} frames. No files created.', flush=True)
        return
    if len(rows_out) != 593 or Counter(r['video'] for r in rows_out) != EXPECTED:
        raise RuntimeError('Not all baseline frames processed')
    if len(worst_images) != 20:
        raise RuntimeError('Expected 20 contact sheets')

    video_summary = summarize(rows_out, ('video',))
    segment_summary = summarize(rows_out, ('video', 'segment'))
    session_summary = summarize(rows_out, ('video', 'job'))
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%d_%H%M%S_%f')
    out = root / 'captures' / ('EXP04_spatial_blur_audit_' + stamp)
    out.mkdir(exist_ok=False)
    (out / 'contact_sheets').mkdir()
    write_csv(out / 'baseline_spatial_blur_per_frame.csv', rows_out)
    write_csv(out / 'baseline_spatial_blur_by_video.csv', video_summary)
    write_csv(out / 'baseline_spatial_blur_by_segment.csv', segment_summary)
    write_csv(out / 'baseline_spatial_blur_by_session.csv', session_summary)
    for job, (ref, rec, result, idx) in sorted(worst_images.items()):
        contact_sheet(out / 'contact_sheets' / (job + '.png'), ref, rec, mask, result, idx)
    (out / 'findings.md').write_text(findings_text(video_summary, segment_summary), encoding='utf-8')
    metadata = {
        'purpose': 'Post-hoc exploratory spatial registration / blur-like attenuation audit',
        'generated_utc': now.isoformat(timespec='seconds'),
        'evaluation_zip': str(evaluation),
        'evaluation_zip_sha256': sha256_file(evaluation),
        'plan_sha256': scoring.PLAN_SHA256,
        'calibration_lock_sha256': scoring.LOCK_SHA256,
        'frozen_scorer_sha256': scoring.FROZEN_SHA256,
        'common_mask_sha256': mask_sha,
        'baseline_sessions': len(by_job), 'baseline_samples': len(rows_out),
        'blur_sigmas_tested': list(SIGMAS),
        'phase_response_min_for_correction': MIN_PHASE_RESPONSE,
        'phase_shift_range_px_for_correction': [MIN_CORRECTABLE_SHIFT, MAX_CORRECTABLE_SHIFT],
        'spatial_correction_mask': 'same 7px-eroded ROI plus 8px image border for before/after',
        'frozen_ssim_psnr_reproduced_for_all_samples': True,
        'barcode_verified_for_all_samples': True,
        'warning': 'Exploratory only. No holdout score or threshold changes. '
                   'Gaussian smoothing is a descriptive blur-like proxy, not codec identification.',
        'script_sha256': sha256_file(Path(__file__)),
    }
    (out / 'audit.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    (out / 'README.txt').write_text(
        'EXP04 SPATIAL / BLUR-LIKE ATTENUATION AUDIT — EXPLORATORY ONLY\n'
        'All 593 eligible baseline images, 20 independent holdout meetings.\n'
        'Source and capture files untouched; original frozen SSIM/PSNR reproduced.\n'
        'Phase correlation estimates XY registration; optional inverse shift is diagnostic.\n'
        'Paired gradient/high-pass ratios describe texture retention, not a codec.\n'
        'Gaussian source smoothing searches 6 candidate sigmas, NOT a new classifier.\n'
        'No Jitsi meetings, no calibration, no oracle prediction changes.\n', encoding='utf-8')
    checks = {p.relative_to(out).as_posix(): sha256_file(p)
              for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'file_checksums.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    archive = out.with_suffix('.zip')
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    print('[COMPLETE] 593 baseline frames / 20 sessions.', flush=True)
    for r in video_summary:
        print(f'  {r["video"]}: median phase shift {r["median_phase_shift_magnitude_px"]:.3f}px; '
              f'gradient ratio {r["median_gradient_ratio"]:.3f}; '
              f'best-blur SSIM gain {r["median_best_blur_ssim_gain"]:+.4f}', flush=True)
    print(f'[ZIP] {archive}', flush=True)
    print('[UNCHANGED] Original PNGs, 240 meetings, calibration lock, frozen oracle.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=r'D:\visual-oracle-revision')
    parser.add_argument('--evaluation', default=None,
                        help='Original EXP04 holdout evaluation ZIP path')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', type=int, default=None,
                        help='Read-only test of N frames per video; no output files')
    args = parser.parse_args()
    if args.smoke is not None and args.smoke < 1:
        parser.error('--smoke must be >=1')
    if args.smoke is not None and args.check_only:
        parser.error('Choose --check-only OR --smoke')
    root = Path(args.root).resolve()
    evaluation = Path(args.evaluation).resolve() if args.evaluation else root / 'captures' / temporal.PKG
    run(root, evaluation, args.check_only, args.smoke)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
