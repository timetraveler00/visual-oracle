#!/usr/bin/env python3
"""Auditable, read-only integrity check for the 405-pair NITS-IQA scores CSV.

This script verifies the exact 9x9x5 filename grid, nine illustrative
I_i D_i L5 pairs, and rank correlations computed from the exported CSV.
It does NOT claim to reproduce the workbook's nonlinear-fitted PLCC/RMSE:
that requires the original calibration function and fitting protocol.

Usage: python nits_iqa_integrity_audit.py NITS_IQA_405_verified.csv
Dependencies: numpy, scipy (already present in visual-oracle environment).
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau

HEADERS = ['Ref','Distorted','DINOv2-Base (%)','DINOv2-Large (%)',
           'Dists (%)','SSIM (%)','OpenCLIP (%)','MOS']
RANK_EXPECTED = {
    'DINOv2-Base (%)': (0.7076,0.5177),
    'DINOv2-Large (%)': (0.6765,0.4894),
    'Dists (%)': (0.7030,0.5165),
    'SSIM (%)': (0.6565,0.4693),
    'OpenCLIP (%)': (0.7297,0.5364),
}
ILLUSTRATIVE = {
    1: (62.40,56.85,62.32,76.60,92.20),
    2: (92.66,86.59,76.75,47.59,89.36),
    3: (95.72,94.91,77.24,66.99,95.59),
    4: (99.43,99.59,96.30,90.99,98.84),
    5: (79.15,79.27,68.02,82.03,91.94),
    6: (84.93,87.35,70.97,49.46,94.34),
    7: (91.84,93.62,82.90,81.73,90.91),
    8: (96.44,96.84,82.06,86.46,91.85),
    9: (92.49,93.42,87.61,89.21,91.72),
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',nargs='?',default=str(Path(__file__).resolve().parent / 'NITS_IQA_405_verified.csv'))
    p.add_argument('--json',help='Optional output JSON path; default: stdout only')
    args=p.parse_args()
    path=Path(args.input)
    raw=path.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=HEADERS:
            raise ValueError(f'Unexpected column headers: {reader.fieldnames!r}')
        rows=list(reader)
    if len(rows)!=405:raise ValueError(f'Expected 405 rows; got {len(rows)}')
    expected={(i,d,l) for i in range(1,10) for d in range(1,10) for l in range(1,6)}
    seen={}
    for line,r in enumerate(rows,2):
        m=re.fullmatch(r'I(\d+)D(\d+)L(\d+)\.bmp',r['Distorted'],flags=re.I)
        if not m:raise ValueError(f'Row {line}: invalid distorted filename')
        key=tuple(map(int,m.groups()))
        if key in seen:raise ValueError(f'Row {line}: duplicate {key}')
        seen[key]=r
        if r['Ref'].casefold()!=f'i{key[0]}.bmp':
            raise ValueError(f'Row {line}: ref/target mismatch')
        for col in HEADERS[2:]:
            try:val=float(r[col])
            except Exception as e:raise ValueError(f'Row {line}: nonnumeric {col}') from e
            if not math.isfinite(val) or not (0<=val<=100):
                raise ValueError(f'Row {line}: out-of-range {col}={val}')
    if set(seen)!=expected:
        raise ValueError(f'Missing={sorted(expected-set(seen))}, extra={sorted(set(seen)-expected)}')
    examples=[]
    for i,vals in ILLUSTRATIVE.items():
        r=seen[(i,i,5)]
        actual=[float(r[c]) for c in HEADERS[2:7]]
        if any(abs(a-b)>0.0051 for a,b in zip(actual,vals)):
            raise ValueError(f'Illustrative pair mismatch: I{i}D{i}L5: {actual} != {vals}')
        examples.append({'pair':i,'reference':r['Ref'],'distorted':r['Distorted'],
                         'mos':float(r['MOS'])})
    mos=[float(r['MOS']) for r in rows]
    corrs=[]
    for col in HEADERS[2:7]:
        v=[float(r[col]) for r in rows]
        rho=float(spearmanr(v,mos).statistic)
        tau=float(kendalltau(v,mos).statistic)
        raw_plcc=float(pearsonr(v,mos).statistic)
        ex_rho,ex_tau=RANK_EXPECTED[col]
        if abs(rho-ex_rho)>0.000055 or abs(tau-ex_tau)>0.000055:
            raise ValueError(f'Rank correlation mismatch: {col}: {rho}, {tau}')
        corrs.append({'metric':col,'spearman_srocc':round(rho,7),
                      'kendall_krocc':round(tau,7),'raw_pearson':round(raw_plcc,7)})
    report={
        'status':'PASS', 'input_filename':path.name,'input_sha256':sha,
        'pairs':405,'references':9,'distortion_codes':9,'severity_levels':5,
        'missing_combinations':0,'duplicate_combinations':0,
        'mos_ge_50':sum(x>=50 for x in mos),'mos_lt_50':sum(x<50 for x in mos),
        'illustrative_nine':examples,'correlations':corrs,
        'limitation':('PLCC/RMSE values stored in the original workbook are not '
                      'raw Pearson/RMSE; original nonlinear mapping and fitting '
                      'code are required for exact reproducibility.'),
        'legacy_360_note':('The 405-row grid is complete. The previous 360/8-category '
                           'statement is not a valid description of this 405-row dataset; '
                           'the historical exclusion provenance remains unverified.'),
    }
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
