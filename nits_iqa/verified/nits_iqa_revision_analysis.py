#!/usr/bin/env python3
"""Revision analyses on the verified 405-pair NITS-IQA table (applsci-4637031, R1).

Place in nits_iqa/verified/ of the visual-oracle repository and run:
    python nits_iqa_revision_analysis.py NITS_IQA_405_verified.csv

Outputs (printed and written as CSV next to the input):
  1. Five-parameter logistic-mapped PLCC and RMSE (VQEG mapping, in-sample).
  2. 95% reference-cluster bootstrap intervals for logistic PLCC
     (600 resamples of the nine reference images, seed 20261108).
  3. SROCC with MOS within each distortion code D1-D9 (45 pairs each).
  4. False-accept / false-reject rates of the Table 4 cut-offs
     against the MOS >= 50 acceptability label.
  5. Reference-cluster bootstrap intervals (4,000 resamples) for raw Pearson
     and SROCC, and paired SROCC differences relative to DINOv2-Base.
  6. LOGO threshold sensitivity (Table 7C): balanced-accuracy threshold per
     training fold, shifted by -5, -2, 0, +2, +5 percentage points.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

METRICS = ["DINOv2-Base (%)", "DINOv2-Large (%)", "Dists (%)", "SSIM (%)", "OpenCLIP (%)"]

def logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5

def fit(s, m, slopes=(0.1, 0.5, 2.0)):
    best = None
    for b2 in slopes:
        try:
            p, _ = curve_fit(logistic5, s, m, p0=[np.ptp(m), b2, np.median(s), 0, np.mean(m)], maxfev=20000)
            pred = logistic5(s, *p)
            rmse = float(np.sqrt(np.mean((pred - m) ** 2)))
            if best is None or rmse < best[1]:
                best = (pearsonr(pred, m)[0], rmse)
        except RuntimeError:
            pass
    return best

def main(path):
    d = pd.read_csv(path)
    d["ref"] = d["Ref"].str.extract(r"I(\d)").astype(int)
    d["code"] = "D" + d["Distorted"].str.extract(r"D(\d)")[0]
    mos = d["MOS"].to_numpy(float)
    out = Path(path).parent

    rng = np.random.default_rng(20261108)
    rows = []
    for k in METRICS:
        s = d[k].to_numpy(float)
        plcc, rmse = fit(s, mos)
        boots = []
        for _ in range(600):
            refs = rng.integers(1, 10, 9)
            idx = np.concatenate([np.where(d["ref"].to_numpy() == r)[0] for r in refs])
            b = fit(s[idx], mos[idx], slopes=(0.5,))
            if b: boots.append(b[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(metric=k, plcc_raw=pearsonr(s, mos)[0], plcc_logistic=plcc,
                         rmse_logistic=rmse, ci_lo=lo, ci_hi=hi,
                         srocc=spearmanr(s, mos)[0]))
    t1 = pd.DataFrame(rows).round(4)
    t1.to_csv(out / "revision_logistic_plcc.csv", index=False); print(t1.to_string(index=False), "\n")

    t2 = d.groupby("code").apply(lambda g: pd.Series({k: spearmanr(g[k], g["MOS"])[0] for k in METRICS})).round(3)
    t2.to_csv(out / "revision_srocc_by_code.csv"); print(t2, "\n")

    y = d["MOS"] >= 50
    rules = {"SSIM>=80": d["SSIM (%)"] >= 80, "SSIM>=75": d["SSIM (%)"] >= 75,
             "DINOv2-Base>=70": d["DINOv2-Base (%)"] >= 70, "DINOv2-Large>=70": d["DINOv2-Large (%)"] >= 70,
             "OpenCLIP>=65": d["OpenCLIP (%)"] >= 65,
             "TC11 (SSIM>=75 & DINOv2-Base>=70 & OpenCLIP>=65)":
                 (d["SSIM (%)"] >= 75) & (d["DINOv2-Base (%)"] >= 70) & (d["OpenCLIP (%)"] >= 65)}
    t3 = pd.DataFrame([dict(rule=r, accepted=int(a.sum()),
                            false_accept=round(100 * (a & ~y).sum() / (~y).sum(), 1),
                            false_reject=round(100 * (~a & y).sum() / y.sum(), 1)) for r, a in rules.items()])
    t3.to_csv(out / "revision_historical_thresholds.csv", index=False); print(t3.to_string(index=False))

def extra(path):
    d = pd.read_csv(path)
    ref = d["Ref"].str.extract(r"I(\d)")[0].astype(int).to_numpy()
    mos = d["MOS"].to_numpy(float)
    out = Path(path).parent
    rng = np.random.default_rng(20261108)
    P = {m: [] for m in METRICS}; S = {m: [] for m in METRICS}; D = {m: [] for m in METRICS[1:]}
    for _ in range(4000):
        idx = np.concatenate([np.where(ref == r)[0] for r in rng.integers(1, 10, 9)])
        base = spearmanr(d[METRICS[0]].to_numpy()[idx], mos[idx])[0]
        for m in METRICS:
            x = d[m].to_numpy()[idx]
            P[m].append(pearsonr(x, mos[idx])[0]); S[m].append(spearmanr(x, mos[idx])[0])
            if m != METRICS[0]: D[m].append(base - S[m][-1])
    t = pd.DataFrame([dict(metric=m, raw_lo=np.percentile(P[m], 2.5), raw_hi=np.percentile(P[m], 97.5),
                           srocc_lo=np.percentile(S[m], 2.5), srocc_hi=np.percentile(S[m], 97.5),
                           diff_vs_dinov2_base=(spearmanr(d[METRICS[0]], mos)[0] - spearmanr(d[m], mos)[0]) if m != METRICS[0] else np.nan,
                           diff_lo=np.percentile(D[m], 2.5) if m != METRICS[0] else np.nan,
                           diff_hi=np.percentile(D[m], 97.5) if m != METRICS[0] else np.nan) for m in METRICS]).round(4)
    t.to_csv(out / "revision_bootstrap_intervals.csv", index=False); print("\n", t.to_string(index=False))

    y = (d["MOS"] >= 50).astype(int).to_numpy(); g = d["Ref"].to_numpy(); rows = []; thr = []
    def bal(yy, pp):
        return 0.5 * (((pp == 1) & (yy == 1)).sum() / (yy == 1).sum() + ((pp == 0) & (yy == 0)).sum() / (yy == 0).sum())
    for m in METRICS:
        s = d[m].to_numpy(float); preds = {sh: np.zeros(len(d), int) for sh in (-5, -2, 0, 2, 5)}
        for r in np.unique(g):
            tr = g != r; best, bt = -1, None
            for c in np.unique(s[tr]):
                sc = bal(y[tr], (s[tr] >= c).astype(int))
                if sc > best: best, bt = sc, c
            thr.append(bt)
            for sh in preds: preds[sh][~tr] = (s[~tr] >= bt + sh).astype(int)
        for sh, p in preds.items():
            rows.append(dict(metric=m, shift=sh, median_threshold=np.median(thr[-9:]), accuracy=(p == y).mean(),
                             false_accept=((p == 1) & (y == 0)).sum() / (y == 0).sum(),
                             false_reject=((p == 0) & (y == 1)).sum() / (y == 1).sum()))
    t = pd.DataFrame(rows).round(4)
    t.to_csv(out / "revision_logo_sensitivity.csv", index=False); print("\n", t.to_string(index=False))

if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "NITS_IQA_405_verified.csv"
    main(f)
    extra(f)
