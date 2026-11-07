#!/usr/bin/env python3
"""Read-only verification of a Git publication snapshot against its SHA-256 manifest.

Uses Git index or HEAD blob bytes, not the Windows working-tree representation.
Does not modify source files, commits, manifests, or experimental records.
"""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.', help='Path to Git repository root')
    p.add_argument('--ref', choices=['index', 'HEAD'], default='index')
    args = p.parse_args()
    failures = []

    def blob(name):
        spec = (':' if args.ref == 'index' else 'HEAD:') + name
        result = subprocess.run(['git', 'show', spec], cwd=args.repo,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(f'Cannot read {name} from Git {args.ref}: '
                               + result.stderr.decode('utf-8', errors='replace')[:200])
        return result.stdout

    try:
        raw = blob('publication_manifest.csv').decode('utf-8-sig')
        rows = list(csv.DictReader(io.StringIO(raw)))
        if len(rows) != 352:
            failures.append(f'Manifest contains {len(rows)} rows, expected 352')
        for row in rows:
            name = row['relative_path']
            b = blob(name)
            actual = hashlib.sha256(b).hexdigest()
            if actual != row['sha256']:
                failures.append(f'SHA-256 mismatch: {name}')
            if len(b) != int(row['bytes']):
                failures.append(f'Byte count mismatch: {name}')
        print(f'[MANIFEST] {len(rows) - sum(x.startswith("SHA-256 mismatch:") for x in failures)} / {len(rows)} SHA-256 matches')

        plan_b = blob('captures/dashboard_jobs/EXP04_main240_plan.json')
        lock_b = blob('captures/dashboard_jobs/EXP04_main240_calibration_lock.json')
        scorer_b = blob('exp04_oracle_frozen.py')
        plan = json.loads(plan_b)
        lock = json.loads(lock_b)
        expected_plan = lock['plan_sha256']
        expected_scorer = lock['scoring_protocol']['frozen_scorer_sha256']
        if hashlib.sha256(plan_b).hexdigest() != expected_plan:
            failures.append('Experiment plan SHA-256 does not match calibration lock')
        if hashlib.sha256(scorer_b).hexdigest() != expected_scorer:
            failures.append('Frozen scorer SHA-256 does not match calibration lock')
        jobs = plan['jobs']
        if len(jobs) != 240:
            failures.append(f'Expected 240 planned sessions, got {len(jobs)}')
        matched = 0
        for job in jobs:
            path = job['relative_job_dir'].replace('\\', '/') + '/scenario.json'
            b = blob(path)
            if hashlib.sha256(b).hexdigest() == job['scenario_sha256']:
                matched += 1
            else:
                failures.append(f'Scenario SHA-256 mismatch: {path}')
        print(f'[SCENARIOS] {matched} / {len(jobs)} match plan')
        print('[LOCK] Plan and scorer hashes checked')

        zip_files = [r['relative_path'] for r in rows if r['relative_path'].lower().endswith('.zip')]
        for name in zip_files:
            with zipfile.ZipFile(io.BytesIO(blob(name))) as nested:
                bad = nested.testzip()
                if bad:
                    failures.append(f'Corrupt nested ZIP: {name}!{bad}')
                for item in nested.namelist():
                    if item.endswith('checksums.json'):
                        checksums = json.loads(nested.read(item))
                        for fname, expected in checksums.items():
                            if hashlib.sha256(nested.read(fname)).hexdigest() != expected:
                                failures.append(f'ZIP checksum mismatch: {name}!{fname}')
        print(f'[ZIP] {len(zip_files)} archives verified (CRC and embedded checksums where present)')
    except Exception as e:
        failures.append(str(e))

    if failures:
        print(f'\n[FAIL] {len(failures)} problems found:')
        for issue in failures[:30]:
            print(' -', issue)
        if len(failures) > 30:
            print(f' ... and {len(failures) - 30} more')
        return 1
    print('\n[PASS] Git publication snapshot matches its manifest and experiment locks.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
