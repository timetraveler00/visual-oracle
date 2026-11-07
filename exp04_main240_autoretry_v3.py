#!/usr/bin/env python3
"""EXP04 main240: preserve and safely replace a measured-but-invalid Jitsi run.

Imports the existing immutable EXP04 plan and the v2 transient-retry logic.
A REVIEW run is eligible ONLY if independent evidence shows a sustained
receiver stream outage (zero RTP decode/bitrate plus failed PNG captures).
Archives original job status, summary and log, marks the raw capture excluded,
and retries in a fresh Jitsi session. Never deletes or edits raw CSV/PNG data.

Windows Anaconda Prompt:
  cd /d D:\\visual-oracle-revision
  python exp04_main240_autoretry_v3.py --check-only
  python exp04_main240_autoretry_v3.py --run calibration

Do not run concurrently with any other EXP04/Jitsi runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import exp04_main240 as core
import exp04_main240_autoretry_v2 as v2

MIN_ZERO_STREAK = 8
MIN_PNG_ERRORS = 8
MIN_SAMPLES = 30
MAX_QUALITY_OK = 20


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def review_eligibility(job):
    """Return (allowed, reason, capture_dir, diagnostics) without writing files."""
    status, state = core.job_status(job)
    if status != 'REVIEW':
        return False, f'Job status {status}, expected REVIEW', None, {}
    if not isinstance(state, dict) or state.get('state') != 'completed':
        return False, 'Status is not a completed run', None, {}
    results = state.get('results', [])
    if state.get('current') != 1 or len(results) != 1:
        return False, 'Ambiguous result count; manual review required', None, {}
    result = results[0]
    if result.get('scenario') != job['condition'] or result.get('severity') != job['severity']:
        return False, 'Scenario or severity mismatch', None, {}
    # Do not retry an otherwise valid scored run just because of a summary issue.
    q = int(result.get('quality_ok') or 0)
    if q > MAX_QUALITY_OK or q >= 28:
        return False, f'{q} quality samples; not the observed severe capture failure', None, {}
    if int(result.get('sample_count') or 0) < MIN_SAMPLES:
        return False, 'Too few total samples to diagnose sustained stream outage', None, {}
    if job['condition'] != 'baseline' and int(result.get('active_samples') or 0) != 0:
        return False, 'Some active fault samples exist; manual review required', None, {}
    folder_value = result.get('run_folder')
    if not isinstance(folder_value, str) or not folder_value:
        return False, 'Missing run_folder', None, {}
    folder = Path(folder_value).resolve()
    capture_root = (core.ROOT / 'captures').resolve()
    if not folder.is_relative_to(capture_root) or not folder.name.startswith('joint_EXP04_'):
        return False, 'Unexpected capture folder path', None, {}
    if not folder.is_dir():
        return False, 'Raw capture folder missing', None, {}
    if (folder / 'INVALID_EXCLUDED_FROM_EXP04_MAIN240.json').exists():
        return False, 'Capture already marked excluded; inspect before proceeding', None, {}
    if not (folder / 'fault_measurements.csv').is_file():
        return False, 'Fault CSV missing; manual review required', None, {}
    csv_path = folder / 'joint_measurements.csv'
    if not csv_path.is_file():
        return False, 'Joint CSV missing', None, {}
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < MIN_SAMPLES:
        return False, f'Joint CSV has only {len(rows)} samples', None, {}
    zero_streak = max_streak = png_errors = zero_count = 0
    for row in rows:
        fps = number(row.get('observed_decode_fps'))
        mbps = number(row.get('bitrate_mbps'))
        if row.get('capture_status') == 'ERROR':
            png_errors += 1
        if fps is not None and mbps is not None and abs(fps) <= 0.01 and abs(mbps) <= 0.001:
            zero_streak += 1
            zero_count += 1
            max_streak = max(max_streak, zero_streak)
        else:
            zero_streak = 0
    details = {'rows': len(rows), 'quality_ok': q, 'png_errors': png_errors,
               'zero_fps_and_bitrate_samples': zero_count,
               'longest_zero_fps_and_bitrate_streak': max_streak,
               'active_samples': result.get('active_samples'),
               'run_valid': result.get('run_valid'),
               'joint_measurements_sha256': sha256(csv_path),
               'fault_measurements_sha256': sha256(folder / 'fault_measurements.csv')}
    if max_streak < MIN_ZERO_STREAK or png_errors < MIN_PNG_ERRORS:
        return False, (f'Not a corroborated stream outage: longest zero streak={max_streak}, '
                       f'PNG errors={png_errors}; manual review required'), folder, details
    jobdir = core.ROOT / job['relative_job_dir']
    log = core.LOGS / (job['name'] + '.log')
    if not log.is_file() or not (jobdir / 'status.json').is_file() and not (jobdir / 'status_recovery.json').is_file():
        return False, 'Job log or status file missing', folder, details
    reason = (f'Sustained receiver stream outage: {max_streak} consecutive zero-FPS/zero-bitrate '
              f'intervals; {png_errors} PNG errors; {q}/{len(rows)} valid images; '
              f'{result.get("active_samples")} active samples')
    return True, reason, folder, details


def archive_review(job, max_retries):
    count = v2.archive_count(job)
    if count >= max_retries:
        raise RuntimeError(f'{job["name"]}: retry budget exhausted ({count}/{max_retries}).')
    ok, reason, folder, details = review_eligibility(job)
    if not ok:
        raise RuntimeError(f'{job["name"]}: REVIEW is not safe to auto-retry: {reason}')
    jobdir = core.ROOT / job['relative_job_dir']
    log = core.LOGS / (job['name'] + '.log')
    archive = v2.ARCHIVE / job['name'] / f'attempt_{count+1:02d}'
    if archive.exists():
        raise FileExistsError(f'Archive already exists: {archive}')
    marker = folder / 'INVALID_EXCLUDED_FROM_EXP04_MAIN240.json'
    record = {
        'job': job['name'], 'stage': job['stage'], 'condition': job['condition'],
        'severity': job['severity'], 'archived_utc': datetime.now(timezone.utc).isoformat(),
        'reason': reason, 'diagnostics': details,
        'original_capture': str(folder), 'archive': str(archive),
        'disposition': 'INVALID EXCLUDED; NOT A COMPLETED EXP04 MAIN240 SESSION',
        'raw_data_preserved': True,
        'analysis_rule': 'Only runs linked by current COMPLETE job status are eligible for analysis; '
                         'do not analyze fault_runs_summary.csv without filtering by valid job runs.',
    }
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    try:
        (archive / 'archive_info.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        for filename in ('status.json', 'status_recovery.json', 'job_summary.csv'):
            src = jobdir / filename
            if src.exists():
                dst = archive / filename
                shutil.move(str(src), str(dst))
                moved.append((src, dst))
        dst_log = archive / log.name
        shutil.move(str(log), str(dst_log))
        moved.append((log, dst_log))
        marker.write_text(json.dumps(record, indent=2), encoding='utf-8', newline='\n')
        if core.job_status(job)[0] != 'PENDING':
            raise RuntimeError('Job did not return to PENDING after archiving; rollback required')
    except Exception:
        marker.unlink(missing_ok=True)
        for src, dst in reversed(moved):
            if dst.exists() and not src.exists():
                shutil.move(str(dst), str(src))
        # Only remove an empty/incomplete archive we just created.
        (archive / 'archive_info.json').unlink(missing_ok=True)
        try:
            archive.rmdir()
        except OSError:
            pass
        raise
    print(f'[REVIEW ARCHIVED] {job["name"]}: {reason}', flush=True)
    print(f'Raw capture retained and marked INVALID: {folder}', flush=True)
    print(f'Original status/log archived: {archive}', flush=True)


def run(stage, max_retries, max_new):
    plan = core.load_plan()
    core.verify_stage_sources(plan, stage)
    if stage == 'holdout':
        core.require_holdout_lock(plan)
    jobs = [j for j in plan['jobs'] if j['stage'] == stage]
    new = 0
    for idx, job in enumerate(jobs, 1):
        status, _ = core.job_status(job)
        if status == 'COMPLETE':
            print(f'[{idx:03d}/120] SKIP completed: {job["name"]}', flush=True)
            continue
        if status == 'REVIEW':
            archive_review(job, max_retries)
            print(f'Waiting {v2.RETRY_WAIT_SECONDS}s before a fresh Jitsi session...', flush=True)
            time.sleep(v2.RETRY_WAIT_SECONDS)
        elif status == 'INTERRUPTED':
            v2.archive_failed(job, max_retries)
            print(f'Waiting {v2.RETRY_WAIT_SECONDS}s before a fresh Jitsi session...', flush=True)
            time.sleep(v2.RETRY_WAIT_SECONDS)
        elif status != 'PENDING':
            raise RuntimeError(f'Unexpected job status {status}: {job["name"]}')
        while True:
            print(f'[{idx:03d}/120] START {job["name"]} ({job["condition"]}/{job["severity"]})', flush=True)
            code, logfile = v2.run_one(job)
            status, _ = core.job_status(job)
            if code == 0 and status == 'COMPLETE':
                print(f'[{idx:03d}/120] COMPLETE {job["name"]}', flush=True)
                new += 1
                break
            print(f'[{idx:03d}/120] FAILED exit={code}, status={status}; log={logfile}', flush=True)
            if status == 'REVIEW':
                archive_review(job, max_retries)
            elif status == 'INTERRUPTED':
                v2.archive_failed(job, max_retries)
            else:
                raise RuntimeError(f'{job["name"]} requires manual review: {status}')
            print(f'Waiting {v2.RETRY_WAIT_SECONDS}s before a fresh Jitsi session...', flush=True)
            time.sleep(v2.RETRY_WAIT_SECONDS)
        if max_new is not None and new >= max_new:
            print(f'Planned stop: {new} new job(s) completed.', flush=True)
            core.print_status(plan)
            return
        if idx < len(jobs):
            time.sleep(core.GAP_SECONDS)
    core.print_status(plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check-only', action='store_true')
    g.add_argument('--run', choices=('calibration', 'holdout'))
    p.add_argument('--max-retries', type=int, default=v2.DEFAULT_MAX_RETRIES)
    p.add_argument('--max-new', type=int)
    args = p.parse_args()
    if args.max_retries < 0 or (args.max_new is not None and args.max_new < 1):
        p.error('--max-retries must be >=0; --max-new must be >=1')
    if args.check_only:
        plan = core.load_plan()
        core.print_status(plan)
        print('REVIEW runs eligible for safe exclusion/retry:')
        count = 0
        for job in plan['jobs']:
            if core.job_status(job)[0] == 'REVIEW':
                ok, reason, folder, details = review_eligibility(job)
                print(f'  {job["name"]}: {"SAFE-TO-RETRY" if ok else "MANUAL REVIEW"}: {reason}')
                count += 1
        if not count:
            print('  None')
    else:
        print('WARNING: Do not run concurrently with other EXP04/Jitsi jobs.', flush=True)
        run(args.run, args.max_retries, args.max_new)


if __name__ == '__main__':
    main()
