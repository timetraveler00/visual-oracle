#!/usr/bin/env python3
"""EXP04: post-hoc temporal alignment sensitivity audit on frozen holdout BASELINES.

For every eligible baseline receiver PNG from Parkrun and ScreenShare, compare
against clean source Y4M at the barcode-identified frame index and +/-1..5
neighboring frames (20..100 ms at 50 FPS). The frozen 640x360 common ROI mask,
SSIM and PSNR implementations are used unchanged. The exact-index scores must
reproduce the immutable holdout evaluation ZIP; any mismatch causes STOP.

This is EXPLORATORY and MUST NOT be used to replace the frozen holdout scores,
reclassify sessions, select thresholds, or claim independent validation of a
new temporal-alignment oracle. Selecting the best offset is optimistic.

No Jitsi meetings, no model loading, no modifications to original captures,
experiment plan, calibration lock, frozen scorer, or scored holdout outputs.

Windows Anaconda Prompt, D:\\visual-oracle-revision:
    python exp04_temporal_alignment_audit.py --check-only
    python exp04_temporal_alignment_audit.py --smoke 2
    python exp04_temporal_alignment_audit.py
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
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

import exp04_score_holdout as scoring
import exp04_deep_calibration as deep

PKG = 'EXP04_holdout_evaluation_20261107_071300.zip'
SAMPLE_CSV = 'EXP04_holdout_5metric_samples.csv'
PRED_CSV = 'EXP04_holdout_frozen_predictions.csv'
FRAME_OFFSETS = tuple(range(-5, 6))
FPS = 50
TARGET = (640, 360)
EXPECTED_BASELINE = {'parkrun': 289, 'screenshare': 304}
SSIM_TOL = 2e-5
PSNR_TOL = 2e-4


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_zip_csv(z, name):
    with z.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')))


def load_verified_evaluation(path):
    with ZipFile(path) as z:
        names = set(z.namelist())
        expected = {SAMPLE_CSV, 'EXP04_holdout_5metric_sessions.csv', PRED_CSV,
                    'EXP04_holdout_frozen_evaluation.json',
                    'EXP04_holdout_scoring_audit.json', 'protocol.json',
                    'EXP04_holdout_checksums.json'}
        if names != expected:
            raise RuntimeError('Unexpected holdout evaluation ZIP contents')
        checks = json.loads(z.read('EXP04_holdout_checksums.json'))
        for name in expected - {'EXP04_holdout_checksums.json'}:
            if hashlib.sha256(z.read(name)).hexdigest() != checks.get(name):
                raise RuntimeError(f'Evaluation ZIP checksum mismatch: {name}')
        audit = json.loads(z.read('EXP04_holdout_scoring_audit.json'))
        if (audit.get('stage') != 'holdout' or audit.get('completed_jobs') != 120 or
            audit.get('eligible_frames') != 3538 or
            audit.get('plan_sha256') != scoring.PLAN_SHA256 or
            audit.get('calibration_lock_sha256') != scoring.LOCK_SHA256 or
            audit.get('frozen_scorer_sha256') != scoring.FROZEN_SHA256):
            raise RuntimeError('Holdout audit/lock mismatch')
        samples = read_zip_csv(z, SAMPLE_CSV)
        preds = read_zip_csv(z, PRED_CSV)
    if len(samples) != 3538 or len(preds) != 120:
        raise RuntimeError('Unexpected evaluation sample/session counts')
    if len({(r['job'], r['sample']) for r in samples}) != 3538:
        raise RuntimeError('Duplicate sample keys in evaluation ZIP')
    baseline = [r for r in samples if r['condition'] == 'baseline']
    if Counter(r['video'] for r in baseline) != EXPECTED_BASELINE:
        raise RuntimeError('Unexpected baseline video/frame counts')
    if (len({r['job'] for r in baseline}) != 20 or
        any(r['severity'] != 'none' or int(r['fault_active']) != 0 for r in baseline)):
        raise RuntimeError('Unexpected baseline labels')
    if len([p for p in preds if p['condition'] == 'baseline']) != 20:
        raise RuntimeError('Expected exactly 20 baseline session predictions')
    return baseline, preds, audit


class SourceCache:
    """Bounded LRU of clean, downscaled Y4M frames; file handle remains open."""
    def __init__(self, source, q, capacity=48):
        self.source = source
        self.q = q
        self.capacity = capacity
        self.cache = OrderedDict()
        self.handle = None

    def __enter__(self):
        self.handle = self.source['path'].open('rb')
        return self

    def __exit__(self, *_):
        if self.handle:
            self.handle.close()
        self.handle = None
        self.cache.clear()

    def frame(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        s = self.source
        frame = self.q.read_reference(self.handle, s['offsets'][idx], s['w'], s['h'], *TARGET)
        self.cache[idx] = frame
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return frame


def best_row(rows, metric):
    """Prefer barcode offset zero on exact ties, then smallest absolute offset."""
    return max(rows, key=lambda r: (r[metric], -abs(r['offset_frames']), -r['offset_frames']))


def compare_one(received, idx, cache, mask, q, reference_ssim, reference_psnr):
    valid = []
    for off in FRAME_OFFSETS:
        candidate = idx + off
        if not 0 <= candidate < len(cache.source['offsets']):
            continue  # Never wrap around at clip boundary.
        clean = cache.frame(candidate)
        ssim, psnr = q.quality_metrics(clean, received, mask)
        if not math.isfinite(ssim) or math.isnan(psnr):
            raise RuntimeError('Invalid SSIM/PSNR in baseline temporal audit')
        # Infinite PSNR is mathematically valid for an exact pixel match.
        valid.append({'offset_frames': off, 'ssim': float(ssim), 'psnr_db': float(psnr)})
    original = next(x for x in valid if x['offset_frames'] == 0)
    if (abs(original['ssim'] - reference_ssim) > SSIM_TOL or
        abs(original['psnr_db'] - reference_psnr) > PSNR_TOL):
        raise RuntimeError(f'Original frozen metric mismatch: '
                           f'{original["ssim"]:.9f}/{original["psnr_db"]:.9f} '
                           f'vs {reference_ssim:.9f}/{reference_psnr:.9f}')
    best_ssim = best_row(valid, 'ssim')
    best_psnr = best_row(valid, 'psnr_db')
    source_ssim, _ = q.quality_metrics(cache.frame(idx),
                                      cache.frame(idx + best_ssim['offset_frames']), mask)
    return valid, original, best_ssim, best_psnr, float(source_ssim)


def panel(image, caption, mask, width=320, height=180):
    out = image.copy()
    out[~mask] = 127
    out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
    label = np.full((45, width, 3), 28, dtype=np.uint8)
    cv2.putText(label, caption[:49], (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .44, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([label, out], axis=0)


def error_panel(ref, rec, caption, mask):
    delta = cv2.absdiff(ref, rec).mean(axis=2)
    heat = cv2.applyColorMap(np.uint8(np.clip(delta * 3, 0, 255)), cv2.COLORMAP_INFERNO)
    return panel(heat, caption, mask)


def write_contact_sheet(path, received, ref0, best_ref, mask, frame_idx, best_off, orig, best):
    images = [
        panel(ref0, f'BARCODE REF frame {frame_idx}', mask),
        panel(best_ref, f'BEST SSIM REF offset {best_off:+d}', mask),
        panel(received, 'RECEIVED (baseline)', mask),
        error_panel(ref0, received, f'ERROR x3 | SSIM {orig:.4f}', mask),
        error_panel(best_ref, received, f'ERROR x3 | SSIM {best:.4f}', mask),
    ]
    if not cv2.imwrite(str(path), np.concatenate(images, axis=1)):
        raise RuntimeError(f'Cannot write contact sheet: {path}')


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize(rows, group_fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] for f in group_fields)].append(r)
    summaries = []
    for key, rr in sorted(groups.items()):
        count = len(rr)
        shifted = [r for r in rr if int(r['best_ssim_offset_frames']) != 0]
        substantive = [r for r in rr if float(r['ssim_gain']) >= 0.03]
        both = [r for r in rr if float(r['ssim_gain']) >= 0.03 and
                float(r['psnr_gain_at_best_ssim_db']) >= 1.0]
        offset_count = Counter(int(r['best_ssim_offset_frames']) for r in rr)
        summary = {f: k for f, k in zip(group_fields, key)}
        summary.update({
            'n': count,
            'median_original_ssim': f'{median([float(r["original_ssim"]) for r in rr]):.9f}',
            'median_best_ssim': f'{median([float(r["best_ssim"]) for r in rr]):.9f}',
            'median_ssim_gain': f'{median([float(r["ssim_gain"]) for r in rr]):.9f}',
            'p90_ssim_gain': f'{float(np.quantile([float(r["ssim_gain"]) for r in rr], .9)):.9f}',
            'median_psnr_gain_at_best_ssim_db': f'{median([float(r["psnr_gain_at_best_ssim_db"]) for r in rr]):.9f}',
            'best_nonzero_n': len(shifted),
            'best_nonzero_percent': f'{100 * len(shifted)/count:.3f}',
            'ssim_gain_ge_0p01_n': sum(float(r['ssim_gain']) >= .01 for r in rr),
            'ssim_gain_ge_0p03_n': len(substantive),
            'gain_ge_0p03_and_psnr_ge_1db_n': len(both),
            'mode_best_ssim_offset': sorted(offset_count, key=lambda off: (-offset_count[off], abs(off), off))[0],
            'best_offset_histogram': ';'.join(f'{off:+d}:{offset_count.get(off, 0)}' for off in FRAME_OFFSETS),
            'median_source_ref_to_best_ref_ssim': f'{median([float(r["source_ssim_at_best_offset"]) for r in rr]):.9f}',
        })
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f'No rows to write: {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_findings(video_summary, segment_summary):
    by_video = {r['video']: r for r in video_summary}
    lines = [
        '# EXP04 post-hoc temporal alignment diagnostic', '',
        '**Exploratory, not a revised holdout test.** Frozen predictions, thresholds, and '
        'original evaluation are unchanged.', '',
        '## Summary by video', '',
        '| Video | Frames | Original median SSIM | Best-offset median SSIM | '
        'Median SSIM gain | Best offset != 0 | Gain >= 0.03 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for video in ('parkrun', 'screenshare'):
        r = by_video[video]
        lines.append(f'| {video} | {r["n"]} | {float(r["median_original_ssim"]):.4f} | '
                     f'{float(r["median_best_ssim"]):.4f} | '
                     f'{float(r["median_ssim_gain"]):+.4f} | '
                     f'{r["best_nonzero_n"]} ({r["best_nonzero_percent"]}%) | '
                     f'{r["ssim_gain_ge_0p03_n"]} |')
    lines.extend(['', '## Temporal segments', '',
                  '| Video | Segment | n | Median SSIM gain | Gain >= 0.03 |',
                  '|---|---|---:|---:|---:|'])
    for r in segment_summary:
        lines.append(f'| {r["video"]} | {r["segment"]} | {r["n"]} | '
                     f'{float(r["median_ssim_gain"]):+.4f} | {r["ssim_gain_ge_0p03_n"]} |')
    lines.extend(['', '## Interpretation cautions', '',
        '- Every received frame carries a barcode-derived source frame index; offset zero is '
        'the frozen evaluation reference and is reproduced exactly.',
        '- Choosing the highest SSIM among up to 11 references is an optimistic post-hoc '
        'selection and **does not demonstrate a valid correction algorithm**.',
        '- Nonzero best offsets can arise from compression, motion blur, interpolated frames, '
        'scene changes, or residual misregistration; they are not proof of temporal lag.',
        '- Static or near-static source frames can make several offsets indistinguishable. '
        'See source_ssim_at_best_offset and the per-offset score table.',
        '- The two videos are the only held-out source contents; samples and sessions are '
        'not independent content realizations.',
        '- The descriptive gain >=0.01 / >=0.03 / PSNR >=1 dB cutoffs are NOT oracle '
        'thresholds, were not calibrated, and must not be used to claim performance gains.',
        '- No classifier predictions or official holdout metrics have been changed.', ''])
    return '\n'.join(lines)


def run(root, evaluation, check_only=False, smoke=None):
    if not root.is_dir() or not evaluation.is_file():
        raise FileNotFoundError(f'Project/evaluation ZIP missing: {root} / {evaluation}')
    os.chdir(root)
    sys.path.insert(0, str(root))
    print('[LOCK] Verifying immutable plan, calibration lock, frozen scorer, and deep module...', flush=True)
    plan, lock = scoring.check_lock(root)
    q = deep.import_quality(root)
    print('[ZIP] Verifying frozen holdout evaluation SHA256 checksums...', flush=True)
    baseline, preds, audit = load_verified_evaluation(evaluation)
    print('[PREFLIGHT] Verifying 120 COMPLETE holdout runs, Y4M hashes and calibration ROI...', flush=True)
    jobs, sources, mask, mask_sha, prepared, exclusions = scoring.prepare_holdout(root, plan, lock, q)
    if mask_sha != audit['common_mask_sha256']:
        raise RuntimeError('ROI mask differs from frozen holdout audit')
    # Verify every baseline sample corresponds exactly to the recorded capture metadata.
    by_job = defaultdict(list)
    for r in baseline:
        by_job[r['job']].append(r)
        if r['job'] not in prepared or r['video'] not in sources:
            raise RuntimeError(f'Unknown baseline job: {r["job"]}')
    for job, rows in by_job.items():
        item = prepared[job]
        if item['job']['condition'] != 'baseline':
            raise RuntimeError(f'Non-baseline original job: {job}')
        by_sample = {int(r['sample']): r for r in item['rows']}
        if set(int(r['sample']) for r in rows) != set(by_sample):
            raise RuntimeError(f'Frozen CSV/capture sample mismatch: {job}')
        for r in rows:
            raw = by_sample[int(r['sample'])]
            if (int(r['frame_index']) != int(raw['frame_index']) or
                int(r['native_width']) != int(float(raw['capture_width'])) or
                int(r['native_height']) != int(float(raw['capture_height'])) or
                int(r['fault_active']) != int(raw['fault_active'])):
                raise RuntimeError(f'Frozen CSV/capture row mismatch: {job}/{r["sample"]}')
    print('[PASS] 593/593 baseline records and 120/120 run statuses verified.', flush=True)
    if check_only:
        print('[CHECK-ONLY] No images processed and no files changed.', flush=True)
        return

    all_per_frame = []
    all_per_offset = []
    worst_images = {}
    for video in ('parkrun', 'screenshare'):
        source_path = sources[video]['path']
        meta_path = source_path.with_name(source_path.stem + '_numbered_metadata.json')
        meta = deep.read_json(meta_path)
        if (int(meta['width']), int(meta['height']), int(meta['frame_count'])) != (1280, 720, 500):
            raise RuntimeError(f'Unexpected barcode metadata: {meta_path}')
        jobs_this_video = sorted(j for j in by_job if prepared[j]['job']['video'] == video)
        print(f'[VIDEO] {video}: {sum(len(by_job[j]) for j in jobs_this_video)} frames, '
              f'{len(jobs_this_video)} baseline sessions.', flush=True)
        processed = 0
        with SourceCache(sources[video], q) as cache:
            for job in jobs_this_video:
                item = prepared[job]
                originals = {int(r['sample']): r for r in item['rows']}
                rows = sorted(by_job[job], key=lambda r: int(r['sample']))
                # Contact sheet shows the worst frozen SSIM frame in each session.
                worst_sample = min(rows, key=lambda r: (float(r['ssim']), int(r['sample'])))['sample']
                for r in rows:
                    if smoke is not None and processed >= smoke:
                        break
                    sample = int(r['sample'])
                    raw = originals[sample]
                    png = item['folder'] / raw['capture_filename']
                    received = cv2.imread(str(png), cv2.IMREAD_COLOR)
                    native = (int(r['native_width']), int(r['native_height']))
                    if received is None or received.shape[1::-1] != native:
                        raise RuntimeError(f'PNG changed/missing: {png}')
                    barcode_index, barcode_confidence, _ = q.decode_barcode(received, meta)
                    idx = int(r['frame_index'])
                    if barcode_index != idx:
                        raise RuntimeError(f'Barcode disagrees with frozen frame index: '
                                           f'{job}/{sample}: {barcode_index} != {idx}')
                    if native != TARGET:
                        received = cv2.resize(received, TARGET, interpolation=cv2.INTER_AREA)
                    comparisons, original, best_ssim, best_psnr, source_ssim = compare_one(
                        received, idx, cache, mask, q, float(r['ssim']), float(r['psnr_db']))
                    ssim_gain = best_ssim['ssim'] - original['ssim']
                    psnr_gain_at_ssim = best_ssim['psnr_db'] - original['psnr_db']
                    segment = '0-8s' if idx < 400 else '8-10s'
                    all_per_frame.append({
                        'job': job, 'video': video, 'block': r['block'],
                        'sample': sample, 'frame_index': idx,
                        'source_time_seconds': f'{idx/FPS:.3f}', 'segment': segment,
                        'native_width': native[0], 'native_height': native[1],
                        'barcode_confidence': f'{barcode_confidence:.3f}',
                        'original_ssim': f'{original["ssim"]:.9f}',
                        'original_psnr_db': f'{original["psnr_db"]:.9f}',
                        'best_ssim_offset_frames': best_ssim['offset_frames'],
                        'best_ssim_offset_ms': best_ssim['offset_frames'] * 1000 // FPS,
                        'best_ssim': f'{best_ssim["ssim"]:.9f}',
                        'ssim_gain': f'{ssim_gain:.9f}',
                        'psnr_at_best_ssim_db': f'{best_ssim["psnr_db"]:.9f}',
                        'psnr_gain_at_best_ssim_db': f'{psnr_gain_at_ssim:.9f}',
                        'best_psnr_offset_frames': best_psnr['offset_frames'],
                        'best_psnr_db': f'{best_psnr["psnr_db"]:.9f}',
                        'best_psnr_gain_db': f'{best_psnr["psnr_db"]-original["psnr_db"]:.9f}',
                        'source_ssim_at_best_offset': f'{source_ssim:.9f}',
                        'offsets_evaluated': len(comparisons),
                    })
                    for c in comparisons:
                        all_per_offset.append({
                            'job': job, 'video': video, 'sample': sample,
                            'frame_index': idx, 'offset_frames': c['offset_frames'],
                            'offset_ms': c['offset_frames'] * 1000 // FPS,
                            'ssim': f'{c["ssim"]:.9f}', 'psnr_db': f'{c["psnr_db"]:.9f}',
                            'ssim_gain_vs_barcode': f'{c["ssim"]-original["ssim"]:+.9f}',
                            'psnr_gain_vs_barcode_db': f'{c["psnr_db"]-original["psnr_db"]:+.9f}',
                        })
                    if smoke is None and str(sample) == str(worst_sample):
                        worst_images[job] = (received.copy(), cache.frame(idx).copy(),
                                             cache.frame(idx + best_ssim['offset_frames']).copy(),
                                             idx, best_ssim['offset_frames'],
                                             original['ssim'], best_ssim['ssim'])
                    processed += 1
                if smoke is not None and processed >= smoke:
                    break
                print(f'  [SESSION] {job}: {len(rows)} frames checked', flush=True)
        if smoke is not None:
            print(f'[SMOKE] {video}: {processed} frames; exact-index scores matched.', flush=True)
    if smoke is not None:
        print(f'[SMOKE COMPLETE] {len(all_per_frame)} frames; no output files created.', flush=True)
        return
    if len(all_per_frame) != 593 or Counter(r['video'] for r in all_per_frame) != EXPECTED_BASELINE:
        raise RuntimeError('Full baseline audit did not cover all 593 frames')
    if len(worst_images) != 20:
        raise RuntimeError(f'Expected 20 contact sheets, got {len(worst_images)}')

    session_summary = summarize(all_per_frame, ('video', 'job'))
    video_summary = summarize(all_per_frame, ('video',))
    segment_summary = summarize(all_per_frame, ('video', 'segment'))
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%d_%H%M%S_%f')
    out = root / 'captures' / ('EXP04_temporal_alignment_audit_' + stamp)
    out.mkdir(parents=False, exist_ok=False)
    (out / 'contact_sheets').mkdir()
    write_csv(out / 'baseline_temporal_per_frame.csv', all_per_frame)
    write_csv(out / 'baseline_temporal_per_offset.csv', all_per_offset)
    write_csv(out / 'baseline_temporal_by_session.csv', session_summary)
    write_csv(out / 'baseline_temporal_by_video.csv', video_summary)
    write_csv(out / 'baseline_temporal_by_segment.csv', segment_summary)
    for job, (received, ref0, refbest, idx, off, orig, best) in sorted(worst_images.items()):
        path = out / 'contact_sheets' / (job + '.png')
        write_contact_sheet(path, received, ref0, refbest, mask, idx, off, orig, best)
    (out / 'findings.md').write_text(make_findings(video_summary, segment_summary), encoding='utf-8')
    (out / 'README.txt').write_text(
        'EXP04 TEMPORAL ALIGNMENT AUDIT (POST-HOC / EXPLORATORY)\n'
        '593 eligible baseline PNGs from 20 independent meetings (10 Parkrun, 10 ScreenShare).\n'
        'For each PNG compare clean source Y4M at barcode frame index and +/-1..5 frames.\n'
        'Offsets are 20 ms at 50 FPS; no circular wraparound.\n'
        'Use exact original 640x360 frozen ROI mask and quality_metrics implementation.\n'
        'Original SSIM/PSNR are verified against the frozen evaluation ZIP.\n'
        'The best offset is an optimistic diagnostic search, NOT a new oracle or corrected score.\n'
        'Twenty contact sheets display the worst SSIM sample from each baseline session.\n'
        'All source/capture/lock/scorer files unchanged; no new Jitsi sessions.\n', encoding='utf-8')
    metadata = {
        'purpose': 'Exploratory post-hoc temporal alignment sensitivity; NOT a revised oracle',
        'generated_utc': now.isoformat(timespec='seconds'),
        'evaluation_zip': str(evaluation), 'evaluation_zip_sha256': sha256_file(evaluation),
        'plan_sha256': scoring.PLAN_SHA256,
        'calibration_lock_sha256': scoring.LOCK_SHA256,
        'frozen_scorer_sha256': scoring.FROZEN_SHA256,
        'common_mask_sha256': mask_sha,
        'baseline_sessions': len(by_job), 'baseline_samples': len(all_per_frame),
        'per_offset_rows': len(all_per_offset), 'offset_frames': list(FRAME_OFFSETS),
        'frame_rate_fps': FPS, 'reference_resolution': list(TARGET),
        'barcode_index_verified_for_all_samples': True,
        'original_ssim_psnr_reproduced_for_all_samples': True,
        'descriptive_cutoffs_not_oracle_thresholds': ['SSIM gain >= 0.01', 'SSIM gain >= 0.03', 'PSNR gain >= 1 dB'],
        'warning': 'Selecting best reference among 11 is optimistic; do not change official holdout predictions.',
        'script_sha256': sha256_file(Path(__file__)),
    }
    (out / 'audit.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    checks = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'file_checksums.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    archive = out.with_suffix('.zip')
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    print('[COMPLETE] All 593 baseline frames, 20 sessions, +/-5 offsets.', flush=True)
    for r in video_summary:
        print(f'  {r["video"]}: median gain {float(r["median_ssim_gain"]):+.5f}; '
              f'gain >= 0.03: {r["ssim_gain_ge_0p03_n"]}/{r["n"]}; '
              f'best nonzero: {r["best_nonzero_n"]}/{r["n"]}', flush=True)
    print(f'[ZIP] {archive}', flush=True)
    print('[UNCHANGED] Original PNGs, all 240 Jitsi runs, frozen thresholds, and predictions.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=r'D:\visual-oracle-revision')
    p.add_argument('--evaluation', default=None, help='Full path to the original holdout evaluation ZIP')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--smoke', type=int, default=None,
                   help='Read-only smoke test: first N eligible baseline frames per video')
    args = p.parse_args()
    if args.smoke is not None and args.smoke < 1:
        p.error('--smoke must be >=1')
    if args.check_only and args.smoke is not None:
        p.error('Use either --check-only or --smoke, not both')
    root = Path(args.root).resolve()
    evaluation = Path(args.evaluation).resolve() if args.evaluation else root / 'captures' / PKG
    run(root, evaluation, args.check_only, args.smoke)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
