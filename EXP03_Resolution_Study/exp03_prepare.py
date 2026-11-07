"""Create a safe, separate EXP03 job for the existing Visual Oracle runner.

No existing capture or job is overwritten. Python 3.10+; standard library only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Prepare EXP03 resolution-sensitivity pilot')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--job-name', default='job_exp03_resolution_pilot_01')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--duration', type=int, default=30)
    ap.add_argument('--interval', type=int, default=2)
    args = ap.parse_args()
    root = args.root.resolve()
    if not 1 <= args.repeats <= 20:
        ap.error('--repeats must be between 1 and 20')
    if args.duration < 5 or args.interval < 1 or args.duration % args.interval:
        ap.error('duration must be >=5 and divisible by interval')
    if not (root / '12_visual_oracle_runner.py').is_file():
        ap.error('Copy this file into D:\\visual-oracle-revision, next to 12_visual_oracle_runner.py')
    if not (root / 'videos' / '720p50_shields_ter.y4m').is_file():
        ap.error('Missing videos\\720p50_shields_ter.y4m')
    jobs = root / 'captures' / 'dashboard_jobs'
    if Path(args.job_name).name != args.job_name or args.job_name in ('.', '..'):
        ap.error('job-name must be a simple directory name')
    target = jobs / args.job_name
    if target.exists():
        ap.error(f'Job already exists; no files changed: {target}')
    scenario = {
        'schema_version': 1,
        'name': 'EXP03 – Resolution Sensitivity Pilot',
        'video': 'videos/720p50_shields_ter.y4m',
        'node_count': 2,
        'duration': args.duration,
        'interval': args.interval,
        'repeats': args.repeats,
        'clip_seconds': 10,
        'perceptual_metrics': [],
        'conditions': [
            {'scenario': 'baseline', 'severity': 'medium'},
            {'scenario': 'blur', 'severity': 'low'},
        ],
        'methodology': (
            'Independent Jitsi sessions, 5s warmup, receiver resolution observed not forced. '
            'Condition order is blocked (all baseline before blur), so time/order is a possible confound. '
            'Offline paired downsampling on the SAME frame pair is a separate controlled '
            'metric-scale sensitivity experiment, not a causal test of WebRTC resolution.'
        ),
    }
    target.mkdir(parents=True, exist_ok=False)
    (target / 'scenario.json').write_text(json.dumps(scenario, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[OK] Created:', target / 'scenario.json')
    print(f'[PLAN] {args.repeats} baseline + {args.repeats} blur-low = {2 * args.repeats} independent sessions')
    print(f'[PLAN] {args.duration // args.interval + 1} samples/session; expected {2 * args.repeats * (args.duration // args.interval + 1)} total')
    print('[RUN] python 12_visual_oracle_runner.py --job captures\\dashboard_jobs\\' + args.job_name)
    print('[NOTE] Do not modify scenario.json after starting the job.')


if __name__ == '__main__':
    main()
