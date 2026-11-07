# NITS-IQA: tabular analysis and reproducibility scope

This folder contains an English-language **publication adaptation** of the
historical NITS-IQA tabular analysis. It is separate from the EXP04 Jitsi study.
The local archival originals have not been modified.

## Contents

- `verified/NITS_IQA_405_verified.csv`: 405 recorded pairs (9 references x
  9 distortion codes x 5 severity levels). The published CSV uses the English
  header `Distorted`; its 405 data records are unchanged.
- `verified/nits_iqa_integrity_audit.py`: verifies the complete filename grid,
  the nine illustrative pairs, MOS split and rank correlations.
- `verified/NITS_IQA_integrity_report.md`: source provenance, recomputed
  correlations and limitations.
- `analysis_notebooks/`: seven English publication adaptations of the
  historical notebooks, with invalid/out-of-order cells repaired, portable CSV
  input and a binary balanced-accuracy implementation equivalent on this dataset.
- `analysis_results/`: derived result tables, including an English-header
  adaptation of the D4/D5 error table.

## Run

Install `numpy`, `pandas`, `scipy`, `scikit-learn`, `jupyter`, and `nbformat`.
From the repository root:

```sh
python nits_iqa/verified/nits_iqa_integrity_audit.py nits_iqa/verified/NITS_IQA_405_verified.csv
jupyter lab
```

Open notebooks from `nits_iqa/analysis_notebooks/`. Each notebook finds the
published CSV relative to the repository or notebook directory. Notebook
outputs were cleared before publication; the analysis code was executed against
the recorded 405-pair table during publication preparation.

## Boundaries

This repository does not include the original BMPs, image scoring pipeline,
original Excel workbook, original model checkpoints, or the historical nonlinear
PLCC/RMSE fitting method. It reproduces the **tabular** analysis, not source
image processing. The workbook's historical fitted PLCC/RMSE values must not be
presented as raw Pearson correlations. The earlier manuscript's 360-image claim
is not supported by the complete 405-row tabular grid.

The original historical notebooks remain only in the local archive. See
`publication_manifest.csv` at the repository root for separate SHA-256 values
of each original and English publication adaptation.
