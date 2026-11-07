#!/usr/bin/env python3
"""EXP04 Shields receiver-side temporal validation: 3 independent paired replications.

Runs the *existing* exp04_temporal_validation.py unchanged, one condition per
fresh Jitsi/Selenium session. Alternates baseline/freeze_low order across pairs,
keeps each session's original raw CSV/report, and produces a session-level
replication report. The earlier exploratory validation is NOT pooled.

Windows Anaconda Prompt:
  cd /d D:\\visual-oracle-revision
  python exp04_temporal_replication.py --check-only
  python exp04_temporal_replication.py

If interrupted, resume the SAME batch, rather than restarting from scratch:
  python exp04_temporal_replication.py --resume "D:\\visual-oracle-revision\\captures\\exp04_temporal_replication_YYYYMMDD_HHMMSS"

No existing experiment script, Y4M, scenario, or prior capture is modified.
Requires a running Jitsi + Selenium Grid for actual runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / 'captures'
VALIDATOR = ROOT / 'exp04_temporal_validation.py'
VERSION = '1.0'


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def check_inputs():
    if not VALIDATOR.is_file():
        raise FileNotFoundError(f'Existing validator not found: {VALIDATOR}')
    proc = subprocess.run([sys.executable, str(VALIDATOR), '--check-only'],
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    print(proc.stdout, end='', flush=True)
    if proc.returncode:
        raise RuntimeError('Original validation preflight failed: ' + proc.stderr.strip())
    if 'INPUT CHECK PASSED' not in proc.stdout:
        raise RuntimeError('Original validator did not confirm INPUT CHECK PASSED')
    paths = {}
    for line in proc.stdout.splitlines():
        for label, key in [('Baseline:', 'baseline'), ('Freeze/Low:', 'freeze_low'),
                           ('Metadata:', 'metadata')]:
            if line.startswith(label):
                paths[key] = Path(line[len(label):].strip()).resolve()
    if set(paths) != {'baseline', 'freeze_low', 'metadata'}:
        raise RuntimeError('Could not read source paths from validator preflight')
    return {'validator_sha256': sha256(VALIDATOR),
            'inputs': {key: fingerprint(path) for key, path in paths.items()}}


def plan(repeats):
    rows = []
    for rep in range(1, repeats + 1):
        order = ('baseline', 'freeze_low') if rep % 2 else ('freeze_low', 'baseline')
        for pos, cond in enumerate(order, 1):
            rows.append({'id': f'R{rep:02d}_{cond}', 'repeat': rep, 'order': pos,
                         'condition': cond, 'status': 'PENDING', 'attempts': []})
    return rows


def check_unchanged(provenance):
    if sha256(VALIDATOR) != provenance['validator_sha256']:
        raise RuntimeError('Original validation script changed after the batch was created')
    for key, old in provenance['inputs'].items():
        current = fingerprint(Path(old['path']))
        if current != old:
            raise RuntimeError(f'{key} source file changed since preflight; do not mix runs')


def safe_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt_number(value, places=4):
    n = safe_number(value)
    return f'{n:.{places}f}' if n is not None else 'N/A'


def median(vals):
    good = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.median(good) if good else None


def session_summary(record, provenance):
    folder = Path(record['output_folder'])
    report_path = folder / 'temporal_validation_report.json'
    if not report_path.is_file():
        raise RuntimeError(f'Missing validation JSON: {report_path}')
    doc = json.loads(report_path.read_text(encoding='utf-8-sig'))
    cond = record['condition']
    session = doc.get('sessions', {}).get(cond)
    summary = doc.get('summaries', {}).get(cond)
    if not session or not summary:
        raise RuntimeError(f'Condition {cond} not found in session/report: {report_path}')
    if session.get('status') != 'COMPLETE':
        raise RuntimeError(f'Incomplete receiver session: {session.get("status")}')
    if Path(session.get('video', '')).resolve() != Path(provenance['inputs'][cond]['path']).resolve():
        raise RuntimeError(f'Session used an unexpected source video: {session.get("video")}')
    if doc.get('fault_window_frames_inclusive') != [150, 224]:
        raise RuntimeError('Unexpected fault window in session report')
    if float(doc.get('source_fps', 0)) != 50.0:
        raise RuntimeError('Unexpected source FPS in session report')
    csv_path = folder / cond / 'receiver_frames.csv'
    if not csv_path.is_file():
        raise RuntimeError(f'Missing raw callback trace: {csv_path}')
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        count = sum(1 for _ in csv.DictReader(f))
    if count != summary.get('callbacks'):
        raise RuntimeError(f'Callback CSV/report mismatch: {count} vs {summary.get("callbacks")}')
    if count != session.get('received_callback_rows'):
        raise RuntimeError('Callback CSV/session count mismatch')
    return {'repeat': record['repeat'], 'order': record['order'], 'condition': cond,
            'output_folder': str(folder), 'callbacks': count,
            'barcode_success_pct': summary.get('barcode_success_pct'),
            'target_pairs': summary.get('target_pairs'),
            'target_median_mae': summary.get('target_median_mae'),
            'other_median_mae': summary.get('other_median_mae'),
            'callback_processing_median_ms': summary.get('callback_processing_median_ms'),
            'substantial_episodes': sum(
                ep.get('duration_s', 0) >= 1.0 and
                ep.get('source_span_frames', 0) >= 40 and
                ep.get('pair_count', 0) >= 10 for ep in summary.get('episodes', [])),
            'substantial_durations_s': [ep['duration_s'] for ep in summary.get('episodes', [])
                if ep.get('duration_s', 0) >= 1.0 and ep.get('source_span_frames', 0) >= 40
                and ep.get('pair_count', 0) >= 10],
            'raw_csv': str(csv_path)}


def evaluate_pair(base, freeze):
    checks = {}
    checks['barcode_95pct'] = all(safe_number(s['barcode_success_pct']) is not None
        and safe_number(s['barcode_success_pct']) >= 95 for s in (base, freeze))
    checks['target_pairs_15'] = all(safe_number(s['target_pairs']) is not None
        and safe_number(s['target_pairs']) >= 15 for s in (base, freeze))
    bm = safe_number(base['target_median_mae'])
    fm = safe_number(freeze['target_median_mae'])
    checks['target_mae_4x'] = (bm is not None and fm is not None and
                               bm >= 4 * max(fm, 0.1))
    checks['two_substantial_freeze_episodes'] = freeze['substantial_episodes'] >= 2
    bo = safe_number(base['other_median_mae'])
    fo = safe_number(freeze['other_median_mae'])
    other_ratio = fo / bo if bo is not None and fo is not None and bo > 0 else None
    checks['outside_mae_comparable'] = (other_ratio is not None and 0.5 <= other_ratio <= 2.0)
    return {'repeat': base['repeat'], 'status': 'PASS' if all(checks.values()) else 'INCONCLUSIVE',
            'baseline_target_median_mae': bm, 'freeze_target_median_mae': fm,
            'freeze_over_baseline_ratio': round(fm / bm, 6) if bm and fm is not None else None,
            'mae_reduction_pct': round(100 * (1 - fm / bm), 3) if bm and fm is not None else None,
            'baseline_outside_median_mae': bo, 'freeze_outside_median_mae': fo,
            'outside_mae_ratio': round(other_ratio, 6) if other_ratio is not None else None,
            'freeze_substantial_episodes': freeze['substantial_episodes'],
            'freeze_substantial_duration_median_s': median(freeze['substantial_durations_s']),
            'baseline_callbacks': base['callbacks'], 'freeze_callbacks': freeze['callbacks'],
            'checks': checks,
            'failed_checks': ', '.join(k for k, v in checks.items() if not v)}


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate(batch, manifest):
    provenance = manifest['provenance']
    summaries = []
    errors = []
    for r in manifest['runs']:
        if r['status'] != 'COMPLETE':
            continue
        try:
            summaries.append(session_summary(r, provenance))
        except Exception as exc:
            errors.append(f'{r["id"]}: {exc}')
    session_fields = ['repeat','order','condition','output_folder','callbacks',
        'barcode_success_pct','target_pairs','target_median_mae','other_median_mae',
        'callback_processing_median_ms','substantial_episodes','substantial_durations_s','raw_csv']
    write_csv(batch / 'replication_sessions.csv', summaries, session_fields)
    pairs = []
    for rep in range(1, manifest['repeats'] + 1):
        conds = {s['condition']: s for s in summaries if s['repeat'] == rep}
        if 'baseline' in conds and 'freeze_low' in conds:
            pairs.append(evaluate_pair(conds['baseline'], conds['freeze_low']))
    pair_fields = ['repeat','status','baseline_target_median_mae','freeze_target_median_mae',
        'freeze_over_baseline_ratio','mae_reduction_pct',
        'baseline_outside_median_mae','freeze_outside_median_mae','outside_mae_ratio',
        'freeze_substantial_episodes','freeze_substantial_duration_median_s',
        'baseline_callbacks','freeze_callbacks','failed_checks']
    write_csv(batch / 'replication_pairs.csv', pairs, pair_fields)
    complete = len(pairs) == manifest['repeats'] and not errors
    if complete and all(p['status'] == 'PASS' for p in pairs):
        verdict = 'REPLICATED INDEPENDENT RECEIVER CONTENT HOLD SUPPORTED'
    elif complete:
        verdict = 'INCONCLUSIVE: AT LEAST ONE INDEPENDENT PAIR DID NOT PASS'
    else:
        verdict = 'INCOMPLETE: INDEPENDENT PAIRED SESSIONS STILL MISSING'
    result = {'verdict': verdict, 'completed_pairs': len(pairs),
              'planned_pairs': manifest['repeats'], 'completed_sessions': len(summaries),
              'planned_sessions': len(manifest['runs']),
              'unit_of_replication': 'fresh baseline+freeze/low Jitsi session pair',
              'not_independent_units': 'repeated 10-second source-video cycles within a session',
              'initial_exploratory_session_included': False,
              'protocol': manifest['protocol'], 'pairs': pairs, 'sessions': summaries,
              'errors': errors, 'source_provenance': provenance}
    save_json(batch / 'replication_report.json', result)
    lines = [
        'EXP04 SHIELDS - INDEPENDENT RECEIVER-SIDE TEMPORAL REPLICATION',
        '=' * 70,
        f'VERDICT: {verdict}',
        f'Independent paired repeats completed: {len(pairs)}/{manifest["repeats"]}',
        f'Fresh Jitsi receiver sessions completed: {len(summaries)}/{len(manifest["runs"])}',
        'The previous exploratory validation is NOT included in these counts.',
        '', 'PREDECLARED PER-PAIR CHECKS:',
        '  1. >=95% valid frame barcodes in each session',
        '  2. >=15 valid advancing target-window pairs in each session',
        '  3. Baseline target median MAE >= 4 * max(freeze target median MAE, 0.1)',
        '  4. >=2 freeze episodes each >=1.0 s, >=40 source-frame advance, >=10 pairs',
        '  5. Freeze/baseline outside-window median MAE ratio within [0.5, 2.0]',
        '', 'PER-PAIR RESULTS:',
    ]
    for p in pairs:
        lines.append(f'  R{p["repeat"]:02d}: {p["status"]}; '
                     f'target MAE baseline={fmt_number(p["baseline_target_median_mae"])}, '
                     f'freeze={fmt_number(p["freeze_target_median_mae"])}; '
                     f'reduction={fmt_number(p["mae_reduction_pct"], 2)}%; '
                     f'outside ratio={p["outside_mae_ratio"]}; '
                     f'substantial freeze episodes={p["freeze_substantial_episodes"]}; '
                     f'median duration={p["freeze_substantial_duration_median_s"]}s')
        if p['failed_checks']:
            lines.append(f'    CHECKS NOT MET: {p["failed_checks"]}')
    if errors:
        lines.extend(['', 'ERRORS:', *('  - ' + x for x in errors)])
    lines.extend(['', 'LIMITATIONS:',
        '  - Independent units are session pairs, NOT within-session video cycles or frame pairs.',
        '  - A source-content freeze is not equivalent to a WebRTC transport freeze.',
        '  - The browser sees presented frames; it cannot recover dropped frames.',
        '  - These new sessions do not retroactively prove the original 33 pilot sessions.',
        '  - This replication does not measure visual-oracle fault-detection accuracy.',
        '', 'Raw session folders are listed in replication_sessions.csv.',
        'Earlier exploratory results are intentionally excluded.', ''])
    (batch / 'replication_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    return result


def run_session(batch, manifest, record):
    check_unchanged(manifest['provenance'])
    attempt_num = len(record['attempts']) + 1
    log_path = batch / f'{record["id"]}_attempt{attempt_num:02d}.log'
    command = [sys.executable, str(VALIDATOR), '--only', record['condition'],
               '--duration', str(manifest['duration']), '--warmup', str(manifest['warmup'])]
    attempt = {'started_at': stamp(), 'log': str(log_path), 'command': command,
               'output_folder': None, 'return_code': None}
    record['attempts'].append(attempt)
    record['status'] = 'RUNNING'
    save_json(batch / 'replication_manifest.json', manifest)
    print(f'\n{"=" * 72}\n{record["id"]}: fresh Jitsi session (attempt {attempt_num})\n{"=" * 72}', flush=True)
    print('NOTE: The original single-condition validator will print INCOMPLETE; '
          'the paired verdict is computed here after both sessions.', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1,
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                match = re.match(r'^Output:\s*(.+?)\s*$', line)
                if match:
                    attempt['output_folder'] = match.group(1)
                    record['output_folder'] = match.group(1)
                    save_json(batch / 'replication_manifest.json', manifest)
            code = proc.wait()
    attempt['return_code'] = code
    attempt['finished_at'] = stamp()
    if code:
        record['status'] = 'FAILED'
        record['last_error'] = f'Validator exited with code {code}; see {log_path}'
        save_json(batch / 'replication_manifest.json', manifest)
        raise RuntimeError(record['last_error'])
    try:
        session_summary(record, manifest['provenance'])
    except Exception as exc:
        record['status'] = 'FAILED'
        record['last_error'] = str(exc)
        save_json(batch / 'replication_manifest.json', manifest)
        raise
    record['status'] = 'COMPLETE'
    record.pop('last_error', None)
    save_json(batch / 'replication_manifest.json', manifest)
    print(f'COMPLETED: {record["id"]}; raw data: {record["output_folder"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='EXP04 independent paired temporal replication')
    parser.add_argument('--repeats', type=int, default=3, help='Fresh independent baseline/freeze session pairs (default 3)')
    parser.add_argument('--duration', type=int, default=35, help='Receiver callback duration per session, seconds (default 35)')
    parser.add_argument('--warmup', type=int, default=5, help='Receiver warmup seconds (default 5)')
    parser.add_argument('--pause', type=int, default=10, help='Seconds between sessions (default 10)')
    parser.add_argument('--check-only', action='store_true', help='Check input files and show the session plan; no Jitsi run')
    parser.add_argument('--resume', type=Path, help='Resume an existing replication batch folder')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 15 <= args.duration <= 180 or args.warmup < 0 or args.pause < 0:
        parser.error('Require 1<=repeats<=10, 15<=duration<=180, warmup>=0, pause>=0')
    if args.resume:
        batch = args.resume.resolve()
        if batch.parent != CAPTURES.resolve():
            parser.error('Resume folder must be directly under the project captures directory')
        manifest_path = batch / 'replication_manifest.json'
        if not manifest_path.is_file():
            parser.error(f'Missing batch manifest: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('version') != VERSION:
            parser.error('Replication script version does not match the batch manifest')
        check_unchanged(manifest['provenance'])
    else:
        provenance = check_inputs()
        manifest = {'version': VERSION, 'created_at': stamp(),
                    'repeats': args.repeats, 'duration': args.duration,
                    'warmup': args.warmup, 'pause': args.pause,
                    'protocol': 'Fresh browser/receiver Jitsi sessions; alternate AB/BA condition order; '
                        '35-second callback observation; no pooling of earlier exploratory data; '
                        'evaluate predeclared five checks per independent pair.',
                    'provenance': provenance, 'runs': plan(args.repeats)}
        if args.check_only:
            print('\nPLANNED FRESH SESSIONS:')
            for r in manifest['runs']:
                print(f'  R{r["repeat"]:02d}, order {r["order"]}: {r["condition"]}')
            print(f'\n{len(manifest["runs"])} sessions, {args.duration}s observation each; '
                  'previous validation excluded. Docker/Jitsi NOT checked by --check-only.')
            return
        CAPTURES.mkdir(parents=True, exist_ok=True)
        batch = CAPTURES / ('exp04_temporal_replication_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        batch.mkdir(exist_ok=False)
        save_json(batch / 'replication_manifest.json', manifest)
    if args.check_only:
        print('RESUME CHECK PASSED:', batch)
        return
    print('\nReplication batch:', batch, flush=True)
    try:
        for index, record in enumerate(manifest['runs']):
            if record['status'] == 'COMPLETE':
                print('Skipping completed session:', record['id'], flush=True)
                continue
            if index and manifest['pause']:
                print(f'Waiting {manifest["pause"]} seconds before the next independent session...', flush=True)
                time.sleep(manifest['pause'])
            run_session(batch, manifest, record)
            write_aggregate(batch, manifest)
    except (KeyboardInterrupt, Exception) as exc:
        write_aggregate(batch, manifest)
        print('\nBatch stopped; completed sessions preserved. Resume with:', flush=True)
        print(f'python exp04_temporal_replication.py --resume "{batch}"', flush=True)
        raise
    result = write_aggregate(batch, manifest)
    print('\n' + '=' * 72, flush=True)
    print('FINAL:', result['verdict'], flush=True)
    print('Report:', batch / 'replication_report.txt', flush=True)
    print('Pairs:', batch / 'replication_pairs.csv', flush=True)
    print('Sessions:', batch / 'replication_sessions.csv', flush=True)
    print('Manifest:', batch / 'replication_manifest.json', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
