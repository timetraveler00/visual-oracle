#!/usr/bin/env python3
"""Exploratory post-hoc threshold analysis of the frozen EXP04 holdout (round-2 review, R2-A).

This analysis uses holdout labels and therefore does NOT change the frozen decision rule
or any reported holdout result. It asks whether a different single-metric threshold, or a
content-specific threshold, could have removed the false alarms.

Input : the holdout evaluation ZIP in captures/ (or the extracted EXP04_holdout_5metric_sessions.csv)
Usage : python captures/exp04_posthoc_threshold_analysis.py captures/EXP04_holdout_evaluation_20261107_071300.zip
Output: EXP04_posthoc_threshold_analysis.csv next to the input
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

METRICS = {  # column, sign (+1: higher = worse), frozen single-metric threshold
    "SSIM q10": ("q10_ssim", -1, 0.81),
    "PSNR q10 (dB)": ("q10_psnr_db", -1, 25.0),
    "DISTS q90": ("q90_dists_distance", 1, 0.1575),
    "DINOv2-Base q10": ("q10_dinov2_base_cosine", -1, 0.9487),
    "OpenCLIP q10": ("q10_openclip_cosine", -1, 0.9399),
}

def youden(bad, y, n_pos, n_neg):
    best = None
    for c in np.unique(bad):
        a = bad >= c
        tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
        j = tp / n_pos - fp / n_neg
        if best is None or j > best[0]:
            best = (j, tp, fp)
    return best[1], best[2]

def load(path):
    if str(path).lower().endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("EXP04_holdout_5metric_sessions.csv"))
            with z.open(name) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)

def main(path):
    s = load(path)
    y = (s["condition"] != "baseline").astype(int).to_numpy()
    rows = []
    for name, (col, sign, thr) in METRICS.items():
        bad = sign * s[col].to_numpy(float)
        frozen = bad > sign * thr
        zero_fp = bad > bad[y == 0].max()          # strictest global threshold with no false alarm
        tpv = fpv = 0
        for v, g in s.groupby("video"):              # content-specific thresholds (upper bound)
            gy = (g["condition"] != "baseline").astype(int).to_numpy()
            tp, fp = youden(sign * g[col].to_numpy(float), gy, int(gy.sum()), int((gy == 0).sum()))
            tpv += tp; fpv += fp
        rows.append({
            "metric": name,
            "AUC pooled": roc_auc_score(y, bad),
            "AUC Parkrun": roc_auc_score(*[(s[s.video == "parkrun"]["condition"] != "baseline").astype(int), sign * s[s.video == "parkrun"][col]]),
            "AUC ScreenShare": roc_auc_score(*[(s[s.video == "screenshare"]["condition"] != "baseline").astype(int), sign * s[s.video == "screenshare"][col]]),
            "frozen TP/FP": f"{int((frozen & (y == 1)).sum())}/{int((frozen & (y == 0)).sum())}",
            "zero-FP global threshold": sign * bad[y == 0].max(),
            "zero-FP TP": int((zero_fp & (y == 1)).sum()),
            "per-video threshold TP/FP": f"{tpv}/{fpv}",
        })
    out = pd.DataFrame(rows).round(4)
    out.to_csv(Path(path).with_name("EXP04_posthoc_threshold_analysis.csv"), index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "captures/EXP04_holdout_evaluation_20261107_071300.zip")
