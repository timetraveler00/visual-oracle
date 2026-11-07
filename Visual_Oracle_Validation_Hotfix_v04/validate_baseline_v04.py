"""Read-only baseline diagnostic for Visual Oracle v0.4.

Run from the project folder:
    python validate_baseline_v04.py
Or:
    python validate_baseline_v04.py --run captures/joint_YYYYMMDD_HHMMSS
"""
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def numeric(x):
    try:
        return float(x) if x not in ('', None) else None
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, help='Specific captures/joint_... directory')
    args = parser.parse_args()
    if args.run:
        folder = args.run.resolve()
    else:
        candidates = sorted((ROOT / 'captures').glob('joint_*/joint_measurements.csv'))
        if not candidates:
            parser.error('No joint_measurements.csv found under captures/joint_*')
        folder = candidates[-1].parent
    path = folder / 'joint_measurements.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error('CSV contains no rows')
    counts = Counter(r.get('stream_liveness') or 'MISSING' for r in rows)
    q_counts = Counter(r.get('quality_status') or 'MISSING' for r in rows)
    play_fps = [numeric(r.get('observed_playback_fps')) for r in rows]
    play_fps = [v for v in play_fps if v is not None]
    post30 = [r for r in rows if (numeric(r.get('elapsed_s')) or 0) >= 35]
    post30_live = sum(r.get('stream_liveness') == 'LIVE' for r in post30)
    changes = sum(str(r.get('stats_stream_changed')).lower() == 'true' for r in rows)
    mismatches = sum(r.get('stats_selection_note') == 'RTP_STATS_MISMATCH_PLAYBACK_ACTIVE' for r in rows)
    print('\nVisual Oracle v0.4 — Baseline diagnostic')
    print('Run:', folder)
    print('Samples:', len(rows))
    print('Capture OK:', sum(r.get('capture_status') == 'OK' for r in rows))
    print('Quality:', dict(q_counts))
    print('Stream liveness:', dict(counts))
    print('Active inbound RTP switches:', changes)
    print('RTP/playback counter mismatches:', mismatches)
    print('Median observed playback FPS:', round(statistics.median(play_fps), 2) if play_fps else 'N/A')
    print('LIVE after 35 s:', f'{post30_live}/{len(post30)}')
    enough = len(rows) >= 11
    live = counts['LIVE'] >= len(rows) - 3 and counts['STALLED'] == 0
    capture = sum(r.get('capture_status') == 'OK' for r in rows) >= len(rows) - 1
    quality = q_counts['OK'] >= len(rows) - 2
    late = len(post30) > 0 and post30_live >= len(post30) - 1
    verdict = 'PASS (initial instrumentation check)' if all([enough, live, capture, quality, late]) else 'REVIEW REQUIRED'
    print('VERDICT:', verdict)
    if not enough:
        print('NOTE: A 60-second / 5-second-interval baseline is recommended.')
    if mismatches:
        print('NOTE: Some RTP stats still disagreed with playback; review raw_stats_*.json.')
    if verdict != 'PASS (initial instrumentation check)':
        print('NOTE: Do not launch the full fault matrix before investigating this run.')
    print('This is an instrumentation smoke test, not scientific validation.')


if __name__ == '__main__':
    main()
