#!/usr/bin/env python3
"""EXP04 main 240-session Jitsi experiment: immutable plan + resumable job orchestration.

Existing 07/08/10/11/12 modules are not edited. Each independent Jitsi session
is a separate 12_visual_oracle_runner.py dashboard job, so a process interruption
never silently reruns already completed sessions.

Usage (Windows Anaconda Prompt, in D:\\visual-oracle-revision):
    python exp04_main240.py --check-only
    python exp04_main240.py --plan
    python exp04_main240.py --status
    python exp04_main240.py --run calibration --max-new 2  # first two jobs only
    python exp04_main240.py --run calibration              # resume remaining jobs
    python exp04_main240.py --run holdout   # requires calibration lock JSON

Do not launch while temporal replication is still running.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'captures' / 'dashboard_jobs'
PLAN = JOBS / 'EXP04_main240_plan.json'
LOCK = JOBS / 'EXP04_main240_calibration_lock.json'
LOGS = ROOT / 'captures' / 'exp04_main240_logs'
RUNNER = ROOT / '12_visual_oracle_runner.py'

VIDEOS = (
    ('shields', 'videos/720p50_shields_ter.y4m', 'calibration'),
    ('fourpeople', 'videos/720p50_fourpeople_exp04.y4m', 'calibration'),
    ('parkrun', 'videos/720p50_parkrun_exp04.y4m', 'holdout'),
    ('screenshare', 'videos/720p50_screenshare_exp04.y4m', 'holdout'),
)
FAULTS = ('blur', 'color', 'freeze', 'black', 'occlusion')
SEVERITIES = ('low', 'medium')
SEED = 20261105
DURATION = 30
INTERVAL = 1
CLIP_SECONDS = 10
BLOCKS = 5
GAP_SECONDS = 10


def render(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def video_header(path, expected_frames):
    with path.open('rb') as f:
        line = f.readline(2048)
    if not line.startswith(b'YUV4MPEG2 ') or not line.endswith(b'\n'):
        raise ValueError(f'Invalid Y4M header: {path}')
    tokens = line.decode('ascii').split()
    params = {v[0]: v[1:] for v in tokens[1:] if len(v) >= 2}
    w, h = int(params['W']), int(params['H'])
    a, b = map(int, params['F'].split(':'))
    if (w, h) != (1280, 720) or (a, b) != (50, 1):
        raise ValueError(f'Expected 1280x720 @ 50 FPS: {path}, got {w}x{h} F{a}:{b}')
    chroma = params.get('C', '420')
    if not chroma.startswith('420') or 'p10' in chroma or 'p12' in chroma:
        raise ValueError(f'Expected 8-bit YUV420: {path}, got {chroma}')
    # Scan frame markers with seeks (not full frame reads): accepts legal FRAME tags.
    frame_bytes = w * h * 3 // 2
    with path.open('rb') as f:
        f.seek(len(line))
        for i in range(expected_frames):
            marker = f.readline(4096)
            if not marker.startswith(b'FRAME') or marker[5:6] not in (b'\n', b'\r', b' '):
                raise ValueError(f'Y4M frame {i} missing or malformed in {path}')
            f.seek(frame_bytes, 1)
        if f.tell() != path.stat().st_size:
            raise ValueError(f'Unexpected trailing data or frame count in {path}: '
                             f'expected {expected_frames} frames')
    return {'width': w, 'height': h, 'fps': 50, 'frames': expected_frames, 'bytes': path.stat().st_size}


def preflight(check_hash=True):
    if not RUNNER.is_file():
        raise FileNotFoundError(f'Missing experiment runner: {RUNNER}')
    for name in ('07_create_numbered_y4m.py', '08_jitsi_synchronized_quality.py',
                 '05_jitsi_video_capture.py', '09_jitsi_webrtc_stats.py',
                 '10_jitsi_joint_quality_performance.py', '11_jitsi_fault_injection_v2.py',
                 '12_perceptual_metrics.py', '12_english_logs.py'):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f'Missing required file: {ROOT / name}')
    out = {}
    for name, rel, stage in VIDEOS:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f'Missing video {name}: {path}. '
                                    'Prepare using exp04_prepare_videos.py (without --force).')
        meta = video_header(path, 504 if name == 'shields' else 500)
        if check_hash:
            meta['sha256'] = sha256(path)
        out[name] = {'path': rel, 'stage': stage, **meta}
    free = shutil.disk_usage(ROOT).free / 1024**3
    print(f'Input videos: Shields 504 frames, other 3 videos 500 frames each; 1280x720 @ 50 FPS; free disk: {free:.1f} GiB')
    if free < 35:
        print('[WARNING] Less than 35 GiB free. Numbered + fault Y4M caches, Docker copies '
              'and 240 capture folders may need substantial additional space.', flush=True)
    return out


def ordered_conditions(video, block):
    conditions = [{'scenario': 'baseline', 'severity': 'medium'},
                  {'scenario': 'baseline', 'severity': 'medium'}]
    conditions += [{'scenario': fault, 'severity': severity}
                   for fault in FAULTS for severity in SEVERITIES]
    seed_bytes = hashlib.sha256(f'{SEED}:{video}:{block}'.encode('utf-8')).digest()
    seed = int.from_bytes(seed_bytes[:8], 'big')
    random.Random(seed).shuffle(conditions)
    return conditions, seed


def build_plan(videos):
    jobs = []
    for name, rel, stage in VIDEOS:
        for block in range(1, BLOCKS + 1):
            conditions, seed = ordered_conditions(name, block)
            for position, condition in enumerate(conditions, 1):
                jobname = f'job_exp04_main_{name}_b{block:02d}_s{position:02d}'
                scenario = {
                    'schema_version': 1,
                    'name': f'EXP04 main {name} block {block:02d} position {position:02d}',
                    'video': rel,
                    'node_count': 2,
                    'duration': DURATION,
                    'interval': INTERVAL,
                    'repeats': 1,
                    'clip_seconds': CLIP_SECONDS,
                    'perceptual_metrics': [],
                    'conditions': [condition],
                    'exp04_phase': 'main240',
                    'exp04_dataset_role': stage,
                    'exp04_video': name,
                    'exp04_block': block,
                    'exp04_position': position,
                    'exp04_shuffle_seed': seed,
                    'exp04_scoring': 'OFFLINE 640x360; no upscaling; same ROI/mask; no test-set threshold tuning',
                }
                jobs.append({
                    'name': jobname,
                    'video': name,
                    'stage': stage,
                    'block': block,
                    'position': position,
                    'condition': condition['scenario'],
                    'severity': 'none' if condition['scenario'] == 'baseline' else condition['severity'],
                    'relative_job_dir': f'captures/dashboard_jobs/{jobname}',
                    'scenario_sha256': sha256_text(render(scenario)),
                    'scenario': scenario,
                })
    assert len(jobs) == 240 and len({j['name'] for j in jobs}) == 240
    for name, _, _ in VIDEOS:
        group = [j for j in jobs if j['video'] == name]
        assert len(group) == 60
        counts = Counter((j['condition'], j['severity']) for j in group)
        assert counts[('baseline', 'none')] == 10
        for fault in FAULTS:
            for severity in SEVERITIES:
                assert counts[(fault, severity)] == 5
        for block in range(1, 6):
            blockjobs = [j for j in group if j['block'] == block]
            assert len(blockjobs) == 12
            assert sum(j['condition'] == 'baseline' for j in blockjobs) == 2
    return {
        'schema_version': 1,
        'experiment': 'EXP04',
        'phase': 'main240',
        'independent_unit': 'one newly started Jitsi meeting per session',
        'session_count': 240,
        'videos': videos,
        'seed': SEED,
        'block_design': '5 blocks/video; each block 2 baseline + 5 faults x 2 severities',
        'duration_seconds': DURATION,
        'sampling_interval_seconds': INTERVAL,
        'clip_seconds': CLIP_SECONDS,
        'evaluation_protocol': {
            'capture': 'Participant_B; existing 10/11/12 pipeline; source-side faults',
            'offline_resolution': '640x360, no upscaling, common ROI/mask',
            'calibration_videos': ['shields', 'fourpeople'],
            'holdout_videos': ['parkrun', 'screenshare'],
            'holdout_policy': 'Do not run holdout until calibration thresholds and analysis rules are locked',
            'pilot_excluded': True,
            'temporal_validation_excluded': True,
        },
        'jobs': jobs,
    }


def public_plan(plan):
    return {**plan, 'jobs': [{k: v for k, v in j.items() if k != 'scenario'} for j in plan['jobs']]}


def create_plan():
    videos = preflight()
    plan = build_plan(videos)
    manifest = public_plan(plan)
    files = [(JOBS / j['name'] / 'scenario.json', render(j['scenario'])) for j in plan['jobs']]
    files.append((PLAN, render(manifest)))
    # Check every target before writing anything; existing differing jobs are never overwritten.
    for path, data in files:
        if path.exists() and path.read_text(encoding='utf-8-sig') != data:
            raise FileExistsError(f'Existing plan differs; refusing to overwrite: {path}')
    for path, data in files:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding='utf-8', newline='\n')
    print('PLAN CREATED / VERIFIED:', PLAN)
    print('240 independent jobs: 120 calibration + 120 holdout')
    print('No Jitsi session was started. No existing job was modified.')


def load_plan():
    if not PLAN.is_file():
        raise FileNotFoundError(f'Plan missing: {PLAN}; run --plan first')
    plan = read_json(PLAN)
    if plan.get('session_count') != 240 or len(plan.get('jobs', [])) != 240:
        raise ValueError('Invalid EXP04 main240 manifest')
    for j in plan['jobs']:
        path = ROOT / j['relative_job_dir'] / 'scenario.json'
        if not path.is_file() or sha256_text(render(read_json(path))) != j['scenario_sha256']:
            raise ValueError(f'Job config missing/changed: {path}')
    return plan


def job_status(job):
    directory = ROOT / job['relative_job_dir']
    paths = [directory / 'status.json', directory / 'status_recovery.json']
    states = []
    for path in paths:
        if path.is_file():
            try:
                state = read_json(path)
                states.append((state.get('updated_at', ''), path.name == 'status_recovery.json', state))
            except Exception:
                pass
    if not states:
        return 'PENDING', None
    state = max(states, key=lambda x: (x[0], x[1]))[2]
    if state.get('state') != 'completed':
        return 'INTERRUPTED', state
    if state.get('current') != 1 or len(state.get('results', [])) != 1:
        return 'REVIEW', state
    result = state['results'][0]
    expected_condition = job['condition']
    expected_severity = job['severity']
    if result.get('scenario') != expected_condition or result.get('severity') != expected_severity:
        return 'REVIEW', state
    folder = Path(result.get('run_folder', ''))
    if not folder.is_dir() or not (folder / 'joint_measurements.csv').is_file() or not (folder / 'fault_measurements.csv').is_file():
        return 'REVIEW', state
    if result.get('run_valid') is not True:
        return 'REVIEW', state
    if int(result.get('quality_ok') or 0) < 28:
        return 'REVIEW', state
    if expected_condition != 'baseline' and int(result.get('active_samples') or 0) < 1:
        return 'REVIEW', state
    return 'COMPLETE', state


def print_status(plan):
    counters = {stage: Counter() for stage in ('calibration', 'holdout')}
    for job in plan['jobs']:
        status, _ = job_status(job)
        counters[job['stage']][status] += 1
    for stage in ('calibration', 'holdout'):
        counts = counters[stage]
        print(f'{stage:12s} complete={counts["COMPLETE"]:3d}/120  '
              f'pending={counts["PENDING"]:3d}  '
              f'interrupted={counts["INTERRUPTED"]:3d}  review={counts["REVIEW"]:3d}')
    print(f'TOTAL: {sum(c["COMPLETE"] for c in counters.values())}/240 completed')


def verify_stage_sources(plan, stage):
    for name, rel, video_stage in VIDEOS:
        if video_stage != stage:
            continue
        path = ROOT / rel
        info = video_header(path, 504 if name == 'shields' else 500)
        saved = plan['videos'][name]
        if info['bytes'] != saved['bytes'] or sha256(path) != saved['sha256']:
            raise RuntimeError(f'Source video changed since the plan was locked: {path}')
    print(f'{stage} video fingerprints verified')


def require_holdout_lock(plan):
    calibration = [j for j in plan['jobs'] if j['stage'] == 'calibration']
    unfinished = [j['name'] for j in calibration if job_status(j)[0] != 'COMPLETE']
    if unfinished:
        raise RuntimeError(f'Calibration not complete: {len(unfinished)} remaining; first: {unfinished[0]}')
    if not LOCK.is_file():
        raise RuntimeError('Holdout is gated. Calibration analysis/thresholds must be locked first. '
                           f'Expected a reviewed lock file: {LOCK}. Do not invent thresholds.')
    lock = read_json(LOCK)
    if lock.get('analysis_locked') is not True or not isinstance(lock.get('thresholds'), dict) or not lock['thresholds']:
        raise RuntimeError('Calibration lock requires analysis_locked=true and non-empty thresholds dictionary')
    if lock.get('plan_sha256') != sha256(PLAN):
        raise RuntimeError('Calibration lock does not match the current plan SHA256')
    if lock.get('calibration_videos') != ['shields', 'fourpeople'] or lock.get('holdout_videos') != ['parkrun', 'screenshare']:
        raise RuntimeError('Calibration lock video roles do not match the preregistered split')
    print('Calibration lock verified:', LOCK)


def run_stage(plan, stage, max_new=None):
    verify_stage_sources(plan, stage)
    if stage == 'holdout':
        require_holdout_lock(plan)
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in plan['jobs'] if j['stage'] == stage]
    newly_completed = 0
    for idx, job in enumerate(jobs, 1):
        status, _ = job_status(job)
        if status == 'COMPLETE':
            print(f'[{idx:03d}/120] SKIP completed: {job["name"]}', flush=True)
            continue
        if status != 'PENDING':
            raise RuntimeError(f'Existing incomplete or questionable job {job["name"]} ({status}). '
                               'Stop and inspect before retrying; automatic reruns are forbidden.')
        print(f'[{idx:03d}/120] START {job["name"]} ({job["condition"]}/{job["severity"]})', flush=True)
        command = [sys.executable, str(RUNNER), '--job', str(ROOT / job['relative_job_dir'])]
        logfile = LOGS / (job['name'] + '.log')
        if logfile.exists():
            raise FileExistsError(f'Existing job log, refusing to overwrite: {logfile}')
        with logfile.open('x', encoding='utf-8', newline='\n') as log:
            log.write(f'UTC start: {datetime.now(timezone.utc).isoformat()}\n')
            log.write(f'Command: {command!r}\n')
            log.flush()
            proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                    errors='replace', bufsize=1)
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
            except KeyboardInterrupt:
                print('\n[STOP] Interrupted. This job must be reviewed before any rerun.', flush=True)
                proc.terminate()
                raise
            log.write(f'Exit code: {code}\n')
        status, _ = job_status(job)
        if code != 0 or status != 'COMPLETE':
            raise RuntimeError(f'Job failed or requires review: {job["name"]}; '
                               f'exit={code}, status={status}, log={logfile}')
        print(f'[{idx:03d}/120] COMPLETE {job["name"]}', flush=True)
        newly_completed += 1
        if max_new is not None and newly_completed >= max_new:
            print(f'Planned stop: {newly_completed} new job(s) completed in this invocation.', flush=True)
            print_status(plan)
            return
        if idx < len(jobs):
            time.sleep(GAP_SECONDS)
    print(f'All {stage} jobs completed and checked.')
    print_status(plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--plan', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--run', choices=['calibration', 'holdout'])
    p.add_argument('--max-new', type=int, default=None, metavar='N',
                   help='With --run: stop cleanly after N new completed jobs; later rerun resumes')
    args = p.parse_args()
    if args.max_new is not None and (args.run is None or args.max_new < 1):
        p.error('--max-new requires --run and N >= 1')
    if args.check_only:
        preflight(check_hash=False)
        plan = build_plan({name: {'path': rel, 'stage': stage} for name, rel, stage in VIDEOS})
        print(f'PLAN CHECK PASSED: {len(plan["jobs"])} jobs; 4 videos x 5 blocks x 12 conditions')
        print('No files written, no Jitsi sessions started.')
    elif args.plan:
        create_plan()
    elif args.status:
        print_status(load_plan())
    elif args.run:
        run_stage(load_plan(), args.run, max_new=args.max_new)


if __name__ == '__main__':
    main()
