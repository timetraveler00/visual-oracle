#!/usr/bin/env python3
"""Inspect and optionally quarantine malformed EXP04 HOLDOUT SCORING JSON checkpoints.

Does not modify EXP04 experiment captures, original sources, plan, calibration
lock, frozen oracle, or existing valid scoring checkpoints. --repair moves only
malformed derived checkpoint JSON and, for a malformed job metadata JSON, its
paired derived score CSV to a dated quarantine folder. All bytes are retained.

Windows Anaconda Prompt:
    python exp04_holdout_json_repair.py
    python exp04_holdout_json_repair.py --repair
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(r'D:\visual-oracle-revision')
SCORING_DIR = Path('captures/exp04_holdout_scoring_v1')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_json(path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding='utf-8-sig'))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--repair', action='store_true', help='Quarantine ONLY malformed derived JSON checkpoints, preserving original bytes')
    args = ap.parse_args()
    root = args.root.resolve()
    base = root / SCORING_DIR
    if not base.is_dir():
        print(f'[NO SCORING FOLDER] {base}')
        print('No derived checkpoint folder to inspect. Do not change experiment files.')
        return
    protocol = base / 'protocol.json'
    jobs = base / 'jobs'
    candidates = ([protocol] if protocol.is_file() else []) + (sorted(jobs.glob('*.json')) if jobs.is_dir() else [])
    bad = [(p, check_json(p)) for p in candidates]
    bad = [(p, error) for p, error in bad if error is not None]
    csvs = sorted(jobs.glob('*.csv')) if jobs.is_dir() else []
    print(f'[SCAN] {base}')
    print(f'       protocol.json: {"present" if protocol.is_file() else "absent"}')
    print(f'       job metadata JSON: {sum(p.parent == jobs for p in candidates)}')
    print(f'       job score CSV: {len(csvs)}')
    print(f'       invalid JSON: {len(bad)}')
    for p, error in bad:
        print(f'[BAD] {p.relative_to(root)} | {p.stat().st_size} bytes | {error}')
    if not bad:
        print('[PASS] No malformed JSON in holdout scoring checkpoints.')
        print('If the original error repeats, share the complete new log for a different diagnosis.')
        return
    if not args.repair:
        print('[NO CHANGES] To preserve all original bytes while isolating ONLY malformed derived checkpoints:')
        print('    python exp04_holdout_json_repair.py --repair')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    quarantine = base / f'quarantine_invalid_json_{stamp}'
    if quarantine.exists():
        raise RuntimeError(f'Quarantine directory already exists: {quarantine}')
    # Identify all files to move before changing anything.
    moves = []
    for p, error in bad:
        if p == protocol:
            moves.append((p, Path('protocol.json'), error))
        elif p.parent == jobs and p.name.startswith('job_exp04_main_'):
            moves.append((p, Path('jobs') / p.name, error))
            paired = p.with_suffix('.csv')
            if paired.is_file():
                moves.append((paired, Path('jobs') / paired.name, f'paired with malformed metadata: {p.name}'))
        else:
            raise RuntimeError(f'Refusing to move unrecognized checkpoint: {p}')
    manifest = {
        'purpose': 'Preserve malformed derived holdout scoring checkpoint bytes for audit',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'original_scoring_folder': str(base),
        'files': [
            {'original_path': str(p), 'quarantine_relative_path': str(rel),
             'bytes': p.stat().st_size, 'sha256': digest(p), 'reason': reason}
            for p, rel, reason in moves
        ],
        'note': 'No experiment captures, lock, model weights, plan, or valid score checkpoints modified.',
    }
    quarantine.mkdir(parents=False, exist_ok=False)
    for p, rel, reason in moves:
        dst = quarantine / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dst)
        print(f'[QUARANTINED] {p.relative_to(root)} -> {dst.relative_to(root)}')
    manifest_path = quarantine / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[REPAIRED] {len(bad)} invalid JSON file(s) isolated; {len(moves)} derived files preserved.')
    print(f'[AUDIT] {manifest_path}')
    print('Now rerun: python exp04_score_holdout.py --max-new 2')
    print('Existing valid checkpoints will be verified by the original script before scoring.')


if __name__ == '__main__':
    main()
