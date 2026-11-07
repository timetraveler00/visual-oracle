#!/usr/bin/env python3
"""EXP04 frozen five-metric session-level oracle (locked before holdout).

Example (Windows Anaconda Prompt, project root):
  python exp04_oracle_frozen.py --stage holdout --input captures/EXP04_holdout_all5_sessions.csv

This script NEVER uses video, condition, severity, stage, or fault_active as
classifier inputs. Those fields are used only to validate/report the experiment.
Each independent Jitsi meeting is one statistical observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_PLAN_SHA = '303dd62dfa915d729b4954bfca3469ecc025167136334d6afb895df27734732f'
EXPECTED_THRESHOLDS = {
    'ssim_q10': ('<', .81),
    'psnr_q10_db': ('<', 25.),
    'dists_q90_distance_blur_branch': ('>', .10),
    'dinov2_q10_cosine_high_blur_branch': ('>', .975),
    'openclip_q10_cosine_high_blur_branch': ('>', .96),
    'dinov2_q10_cosine_low_semantic_branch': ('<', .94),
    'openclip_q10_cosine_low_semantic_branch': ('<', .93),
    'dists_q90_distance_single': ('>', .1575),
    'dinov2_q10_cosine_single': ('<', .9487),
    'openclip_q10_cosine_single': ('<', .9399),
}
SCORE_KEYS = ('q10_ssim', 'q10_psnr_db', 'q90_dists_distance',
              'q10_dinov2_base_cosine', 'q10_openclip_cosine')
PRED_KEYS = ('SSIM', 'PSNR', 'DISTS', 'DINOv2', 'OpenCLIP',
             'SSIM_AND_PSNR', 'SSIM_OR_PSNR', 'ALL5_AND', 'ALL5_OR', 'LAYERED_PRIMARY')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_lock(lock, lock_path, plan_path):
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA:
        raise RuntimeError('Experiment plan SHA256 mismatch; stop.')
    if lock.get('analysis_locked') is not True or lock.get('holdout_authorized') is not True:
        raise RuntimeError('EXP04 calibration lock is not active; stop.')
    if lock.get('plan_sha256') != EXPECTED_PLAN_SHA:
        raise RuntimeError('Calibration lock does not match experiment plan; stop.')
    if lock.get('calibration_videos') != ['shields', 'fourpeople'] or lock.get('holdout_videos') != ['parkrun', 'screenshare']:
        raise RuntimeError('Dataset roles changed; stop.')
    if lock.get('scoring_protocol', {}).get('frozen_scorer_sha256') != sha256_file(__file__):
        raise RuntimeError('Frozen scorer script SHA256 mismatch; stop.')
    if set(lock.get('thresholds', {})) != set(EXPECTED_THRESHOLDS):
        raise RuntimeError('Threshold set changed; stop.')
    for key, (op, value) in EXPECTED_THRESHOLDS.items():
        item = lock['thresholds'][key]
        if item['operator'] != op or float(item['value']) != value:
            raise RuntimeError(f'Locked threshold changed: {key}; stop.')


def classify(scores):
    """ONLY the five numeric metric summaries are used to classify."""
    s, p, d, n, c = (float(scores[k]) for k in SCORE_KEYS)
    if not all(map(math.isfinite, (s, p, d, n, c))):
        raise ValueError('Nonfinite metric in session')
    singles = {
        'SSIM': s < .81, 'PSNR': p < 25., 'DISTS': d > .1575,
        'DINOv2': n < .9487, 'OpenCLIP': c < .9399,
    }
    l1 = singles['SSIM'] or singles['PSNR']
    l2 = d > .10 and n > .975 and c > .96
    l3 = n < .94 and c < .93
    out = {**singles,
           'SSIM_AND_PSNR': singles['SSIM'] and singles['PSNR'],
           'SSIM_OR_PSNR': l1,
           'ALL5_AND': all(singles.values()),
           'ALL5_OR': any(singles.values()),
           'LAYERED_PRIMARY': l1 or l2 or l3}
    return (l1, l2, l3), out


def wilson(success, total, z=1.959963984540054):
    if total == 0:
        return None
    phat = success / total
    a = 1 + z*z/total
    center = (phat + z*z/(2*total)) / a
    radius = z*math.sqrt(phat*(1-phat)/total + z*z/(4*total*total)) / a
    return [round(max(0., center-radius), 6), round(min(1., center+radius), 6)]


def exact_mcnemar(b, c):
    n = b+c
    if n == 0:
        return 1.
    # Exact two-sided binomial p-value under the paired null, capped at 1.
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(b, c)+1)) / (2**n))


def confusion(rows, key):
    tp = sum(r['actual_fault'] == 1 and r['predictions'][key] for r in rows)
    tn = sum(r['actual_fault'] == 0 and not r['predictions'][key] for r in rows)
    fp = sum(r['actual_fault'] == 0 and r['predictions'][key] for r in rows)
    fn = sum(r['actual_fault'] == 1 and not r['predictions'][key] for r in rows)
    ratio = lambda a, b: a / b if b else None
    precision, recall = ratio(tp, tp+fp), ratio(tp, tp+fn)
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, accuracy=ratio(tp+tn, len(rows)),
                precision=precision, recall=recall, specificity=ratio(tn, tn+fp),
                F1=f1, FPR=ratio(fp, fp+tn), FNR=ratio(fn, fn+tp),
                recall_wilson95=wilson(tp, tp+fn), specificity_wilson95=wilson(tn, tn+fp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('calibration', 'holdout'))
    parser.add_argument('--input', type=Path, help='One row per session, with five numeric metrics')
    parser.add_argument('--verify-lock-only', action='store_true', help='Check plan, lock, and scorer SHA256; no data read')
    parser.add_argument('--plan', type=Path, default=Path('captures/dashboard_jobs/EXP04_main240_plan.json'))
    parser.add_argument('--lock', type=Path, default=Path('captures/dashboard_jobs/EXP04_main240_calibration_lock.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('captures'))
    args = parser.parse_args()
    lock, plan = load_json(args.lock), load_json(args.plan)
    check_lock(lock, args.lock, args.plan)
    if args.verify_lock_only:
        print('[PASS] Plan, active lock, frozen scorer SHA256 and thresholds verified.')
        print('       Plan SHA256:', EXPECTED_PLAN_SHA)
        print('       Lock SHA256:', sha256_file(args.lock))
        print('       Scorer SHA256:', sha256_file(__file__))
        print('       No experiments started and no files changed.')
        return
    if not args.stage or args.input is None:
        parser.error('--stage and --input are required unless --verify-lock-only is used')
    jobs = {j['name']: j for j in plan['jobs'] if j['stage'] == args.stage}
    if len(jobs) != 120:
        raise ValueError('Plan does not contain exactly 120 jobs for stage')
    with args.input.open(encoding='utf-8-sig', newline='') as f:
        input_rows = list(csv.DictReader(f))
    if len(input_rows) != 120 or {r['job'] for r in input_rows} != set(jobs):
        raise ValueError('Input must contain exactly the 120 unique planned sessions for selected stage')
    rows = []
    for r in input_rows:
        j = jobs[r['job']]
        for key in ('video', 'condition', 'severity'):
            if r[key] != j[key]:
                raise ValueError(f'Session metadata mismatch: {j["name"]}/{key}')
        if int(r['block']) != j['block']:
            raise ValueError(f'Block mismatch: {j["name"]}')
        if int(r['n']) < int(lock['scoring_protocol']['minimum_eligible_frames_per_session']):
            raise ValueError(f'Too few eligible frames: {j["name"]}; exclude and report separately, do not impute')
        layers, predictions = classify(r)  # Metadata is deliberately NOT passed.
        rows.append({'source': r, 'actual_fault': int(j['condition'] != 'baseline'),
                     'layers': layers, 'predictions': predictions})
    by_name = {r['source']['job']: r for r in rows}
    rows = [by_name[j] for j in jobs]
    summary = {key: confusion(rows, key) for key in PRED_KEYS}
    def by_group(group_key):
        groups = defaultdict(list)
        for row in rows:
            groups[row['source'][group_key]].append(row)
        return {k: {'sessions': len(v), 'layered': confusion(v, 'LAYERED_PRIMARY'),
                    'structural_or': confusion(v, 'SSIM_OR_PSNR')}
                for k, v in sorted(groups.items())}
    better = sum(r['predictions']['LAYERED_PRIMARY'] == r['actual_fault'] and
                 r['predictions']['SSIM_OR_PSNR'] != r['actual_fault'] for r in rows)
    worse = sum(r['predictions']['LAYERED_PRIMARY'] != r['actual_fault'] and
                r['predictions']['SSIM_OR_PSNR'] == r['actual_fault'] for r in rows)
    audit = {
        'stage': args.stage, 'sessions': len(rows),
        'plan_sha256': sha256_file(args.plan), 'lock_sha256': sha256_file(args.lock),
        'frozen_scorer_sha256': sha256_file(__file__), 'input_sessions_sha256': sha256_file(args.input),
        'results': summary, 'by_video': by_group('video'),
        'by_condition': by_group('condition'), 'by_severity': by_group('severity'),
        'layer_activation_counts': dict(Counter(k for r in rows for k, active in zip(('layer1','layer2','layer3'),r['layers']) if active)),
        'paired_layered_vs_structural_or': {
            'improved': better, 'worsened': worse,
            'exact_mcnemar_two_sided_p': exact_mcnemar(better, worse)},
        'caution': 'Calibration figures are apparent/in-sample, not independent validation' if args.stage=='calibration' else 'Holdout thresholds frozen before any holdout scoring',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f'EXP04_{args.stage}_frozen_predictions.csv'
    out_json = args.output_dir / f'EXP04_{args.stage}_frozen_evaluation.json'
    fields = ['job','video','block','condition','severity','n',*SCORE_KEYS,'actual_fault',
              'layer_1_structural','layer_2_perceptual_semantic_consistency','layer_3_semantic_break',
              *(f'predict_{key}' for key in PRED_KEYS)]
    with out_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for item in rows:
            r = item['source']; l1,l2,l3 = item['layers']
            writer.writerow({**{k:r[k] for k in ['job','video','block','condition','severity','n',*SCORE_KEYS]},
                             'actual_fault': item['actual_fault'],
                             'layer_1_structural': int(l1),
                             'layer_2_perceptual_semantic_consistency': int(l2),
                             'layer_3_semantic_break': int(l3),
                             **{f'predict_{key}': int(item['predictions'][key]) for key in PRED_KEYS}})
    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    print(f'[PASS] {args.stage}: {len(rows)} independent sessions')
    for name in ('SSIM_OR_PSNR','LAYERED_PRIMARY'):
        r = summary[name]
        print(f'  {name}: TP={r["TP"]} FN={r["FN"]} FP={r["FP"]} TN={r["TN"]} F1={r["F1"]:.4f}')
    print('  McNemar exact p:',audit['paired_layered_vs_structural_or']['exact_mcnemar_two_sided_p'])
    print('  Saved:',out_csv, 'and', out_json)


if __name__ == '__main__':
    main()
