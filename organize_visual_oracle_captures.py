#!/usr/bin/env python3
"""Index Visual Oracle capture runs and create descriptive Windows junction aliases.

Does not rename, move, or duplicate any capture folder or media file.
Uses dashboard_jobs/*/status.json and job_summary.csv as source of truth.

Usage:
  python organize_visual_oracle_captures.py --dry-run
  python organize_visual_oracle_captures.py
  python organize_visual_oracle_captures.py --no-junctions
  python organize_visual_oracle_captures.py --job job_exp02_pilot_01
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HEADER = 'VISUAL_ORACLE_CAPTURE_INFO_V1'
FIELDS = ('experiment', 'job', 'plan_index', 'scenario', 'severity', 'repeat_index',
          'original_folder', 'alias_folder', 'samples', 'quality_ok', 'run_valid')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_name(value, limit=80):
    value = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value or 'unknown')).strip('_')
    return (value or 'unknown')[:limit]


def under(path: Path, parent: Path):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover(job_dir: Path):
    scenario_file = job_dir / 'scenario.json'
    if not scenario_file.is_file():
        return [], 'missing scenario.json'
    scenario = read_json(scenario_file)
    experiment = str(scenario.get('name') or job_dir.name)
    status_file = job_dir / 'status.json'
    summary_file = job_dir / 'job_summary.csv'
    rows = []
    if status_file.is_file():
        status = read_json(status_file)
        if isinstance(status.get('results'), list):
            rows = [r for r in status['results'] if isinstance(r, dict)]
    if not rows and summary_file.is_file():
        with summary_file.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        return [], 'no completed runs in status.json/job_summary.csv'
    counts = {}
    result = []
    for pos, row in enumerate(rows, 1):
        original = row.get('run_folder')
        if not original:
            continue
        scenario_name = str(row.get('scenario') or 'unknown').lower()
        severity = str(row.get('severity') or 'none').lower()
        if scenario_name == 'baseline':
            severity = 'none'
        key = (scenario_name, severity)
        counts[key] = counts.get(key, 0) + 1
        try:
            rep = int(row.get('repeat_index') or counts[key])
        except (ValueError, TypeError):
            rep = counts[key]
        try:
            plan = int(row.get('plan_index') or pos)
        except (ValueError, TypeError):
            plan = pos
        result.append(dict(experiment=experiment, job=job_dir.name,
                           plan_index=plan, scenario=scenario_name,
                           severity=severity, repeat_index=rep,
                           original_folder=str(original),
                           samples=row.get('sample_count', ''),
                           quality_ok=row.get('quality_ok', ''),
                           run_valid=row.get('run_valid', '')))
    return result, ''


def write_info(folder: Path, row: dict, dry_run: bool):
    path = folder / 'CAPTURE_INFO.txt'
    if path.exists():
        try:
            first_line = path.read_text(encoding='utf-8-sig').splitlines()[0]
        except (OSError, IndexError):
            first_line = ''
        if first_line != HEADER:
            print(f'[SKIP INFO] Existing non-tool file: {path}')
            return
    content = (f'{HEADER}\n'
               f'Experiment : {row["experiment"]}\n'
               f'Job        : {row["job"]}\n'
               f'Condition  : {row["scenario"]}/{row["severity"]}\n'
               f'Plan index : {row["plan_index"]}\n'
               f'Repeat     : {row["repeat_index"]}\n'
               f'Original   : {row["original_folder"]}\n'
               f'Alias      : {row["alias_folder"]}\n'
               f'Samples    : {row["samples"]}\n'
               f'Quality OK : {row["quality_ok"]}\n'
               f'Run valid  : {row["run_valid"]}\n')
    if not dry_run:
        path.write_text(content, encoding='utf-8')


def make_junction(alias: Path, target: Path, dry_run: bool):
    if alias.exists() or alias.is_symlink():
        if alias.resolve() == target.resolve():
            return 'EXISTS'
        return 'CONFLICT'
    if dry_run:
        return 'WOULD CREATE'
    alias.parent.mkdir(parents=True, exist_ok=True)
    # Windows directory junctions do not duplicate files or require Developer Mode.
    proc = subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(target)],
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        return f'FAILED: {proc.stderr.strip() or proc.stdout.strip()}'
    return 'CREATED'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='D:\\visual-oracle-revision (defaults to script folder)')
    parser.add_argument('--job', help='Only one dashboard job folder name')
    parser.add_argument('--dry-run', action='store_true', help='Preview without writing anything')
    parser.add_argument('--no-junctions', action='store_true', help='Write index and info only')
    args = parser.parse_args()
    root = args.root.resolve()
    captures = root / 'captures'
    jobs = captures / 'dashboard_jobs'
    if not jobs.is_dir():
        parser.error(f'Not found: {jobs}. Place this script in the experiment root or use --root.')
    if not args.no_junctions and os.name != 'nt' and not args.dry_run:
        parser.error('Junction creation requires Windows; use --no-junctions on this OS.')
    output = captures / 'by_experiment'
    candidates = [jobs / args.job] if args.job else sorted(p for p in jobs.iterdir() if p.is_dir())
    entries = []
    for job_dir in candidates:
        if not job_dir.is_dir():
            print(f'[SKIP JOB] Not found: {job_dir}')
            continue
        rows, reason = discover(job_dir)
        if reason:
            print(f'[SKIP JOB] {job_dir.name}: {reason}')
            continue
        for row in rows:
            target = Path(row['original_folder'])
            if not target.is_absolute():
                target = (root / target).absolute()
            # Never follow a status entry outside the captures tree.
            if not under(target, captures) or under(target, jobs) or under(target, output):
                print(f'[SKIP RUN] Unsafe capture path: {target}')
                continue
            if not target.is_dir():
                print(f'[SKIP RUN] Missing capture directory: {target}')
                continue
            experiment_slug = safe_name(row['experiment'], 64)
            alias_name = (f'{row["plan_index"]:02d}_'
                          f'{safe_name(row["scenario"], 20)}_'
                          f'{safe_name(row["severity"], 16)}_'
                          f'R{row["repeat_index"]:02d}__{safe_name(target.name, 36)}')
            alias = output / experiment_slug / alias_name
            row['alias_folder'] = str(alias)
            status = 'INDEX ONLY' if args.no_junctions else make_junction(alias, target, args.dry_run)
            if status == 'CONFLICT' or status.startswith('FAILED'):
                print(f'[WARN] {status}: {alias}')
                row['alias_folder'] = ''
            else:
                print(f'[{status}] {experiment_slug} / {alias_name}')
            write_info(target, row, args.dry_run)
            entries.append(row)
    if not entries:
        print('No runs found. Nothing changed.')
        return 1
    index = output / 'CAPTURE_INDEX.csv'
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        # With --job, retain rows for other jobs from a pre-existing index.
        if args.job and index.exists():
            with index.open(encoding='utf-8-sig', newline='') as handle:
                old = list(csv.DictReader(handle))
            entries = [r for r in old if r.get('job') != args.job] + entries
        with index.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(entries)
    print(f'\n{len(entries)} indexed runs. {"Would write" if args.dry_run else "Index written"}: {index}')
    print('Original capture directories and media files were not moved or renamed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
