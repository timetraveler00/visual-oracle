# NITS-IQA — data integrity and reproducibility audit (2026-11-07)

## Source and checksums

- Source: `image_quality_results.xlsx`, comparison worksheet, range `A1:H406` (405 data rows), and correlation worksheet, range `A1:E6`.
- Original workbook SHA-256: `ef867d38affef48c8160df0df6e48e80a8856fce4536c415bae52e7f72978eaa`.
- Lossless tabular export (values): `NITS_IQA_405_verified.csv`, SHA-256: `ffde705124a249cb933be0e360a13927c156e239b8697b787d47d1c5e701f279`.
- The accompanying `nits_iqa_integrity_audit.py` checks the CSV independently and produces a JSON report.

## Verified dataset structure

| Property | Verified value |
|---|---:|
| Reference images | 9 (`I1.bmp`–`I9.bmp`) |
| Distortion codes | 9 (`D1`–`D9`) |
| Severity levels | 5 (`L1`–`L5`) |
| Expected pairs | 9 × 9 × 5 = 405 |
| Actual pairs | **405** |
| Missing/duplicate `(reference, distortion, level)` tuples | **0/0** |
| Invalid filename-to-reference associations | **0** |
| Missing, nonfinite or out-of-range metric/MOS entries | **0** |
| MOS range | 6–90 |
| MOS ≥ 50 | 199 |
| MOS < 50 | 206 |

Each reference appears 45 times; each distortion code appears 45 times; each severity level appears 81 times.

## Original nine illustrative pairs: exact identification

The nine rows of Table 5 in the pre-revision manuscript are **exactly** the following entries. Each five-score vector matches the original manuscript to two decimals, with a unique matching record in the workbook:

| Manuscript pair | Reference | Distorted image | MOS |
|---:|---|---|---:|
| 1 | I1.bmp | I1D1L5.bmp | 19 |
| 2 | I2.bmp | I2D2L5.bmp | 16 |
| 3 | I3.bmp | I3D3L5.bmp | 28 |
| 4 | I4.bmp | I4D4L5.bmp | 22 |
| 5 | I5.bmp | I5D5L5.bmp | 14 |
| 6 | I6.bmp | I6D6L5.bmp | 14 |
| 7 | I7.bmp | I7D7L5.bmp | 13 |
| 8 | I8.bmp | I8D8L5.bmp | 27 |
| 9 | I9.bmp | I9D9L5.bmp | 24 |

**Selection rule:** `I_i D_i L5` for `i = 1..9`. This is a diagonal illustrative selection covering all nine reference IDs and all nine distortion codes at the most severe numbered level; it is not random and does not represent the severity distribution of the full dataset.

## Recomputed correlations on all 405 pairs

| Metric | SROCC (recomputed) | KROCC (recomputed) | Raw Pearson PLCC (recomputed) | PLCC stored in workbook |
|---|---:|---:|---:|---:|
| DINOv2-Base | 0.7076 | 0.5177 | 0.4995 | 0.6909 |
| DINOv2-Large | 0.6765 | 0.4894 | 0.5123 | 0.6650 |
| DISTS similarity (%) | 0.7030 | 0.5165 | 0.6780 | 0.7027 |
| SSIM | 0.6565 | 0.4693 | 0.5665 | 0.6416 |
| OpenCLIP | 0.7297 | 0.5364 | 0.6488 | 0.7159 |

All five SROCC and KROCC values reproduce the `Korelasyon` sheet to four decimal places. **The stored PLCC values do not equal the ordinary Pearson correlations of the 405 displayed metric scores and MOS.** They may have been computed after a nonlinear quality-score mapping, but the workbook contains only final values, not the mapping formula, fitting method, or fitted parameters. The historical PLCC/RMSE figures are therefore **not independently reproducible from this workbook alone**. Do not relabel them as raw Pearson correlations.

A defensible reproducible alternative is to report the raw Pearson PLCC values above, explicitly labeled **uncalibrated Pearson correlation**, and to distinguish them from any previously reported fitted PLCC values. If the original mapping script becomes available, the published fitted PLCC and RMSE can be audited and retained with full methodological detail.

## The 360/405 issue

The original manuscript mentions 360 distorted images and eight distortion categories, whereas the present verified workbook contains all 405 combinations across nine distortion codes. The difference of 45 records is arithmetically one distortion code across nine references and five severity levels, but **there is no evidence here identifying a specific excluded category or confirming that the earlier 360-image configuration was an actual filtered subset**. Therefore do not assert that `D9` was excluded, and do not present 360 as the denominator for the verified full-dataset analysis. Preserve the earlier description only as historical context if its provenance can be demonstrated; otherwise correct it in the revision.

## Remaining provenance limitations

1. The workbook does not contain the original BMP image bytes, image-generation metadata, or MOS source sheets. This audit verifies **tabular integrity**, not source-image authenticity.
2. The names `D1`–`D9` are coded identifiers; the exact mapping to distortion-category names must be verified against the source dataset documentation before publishing a per-distortion table.
3. `Dists (%)` is a high-is-better tabular similarity percentage. Do **not** confuse it with the native low-is-better DISTS distance used in EXP04.
4. The five score columns are rounded to two decimal places. Recomputed correlations are from these recorded values; minor deviations from unrounded source outputs are possible.
5. Full replication of the original deep-model scoring requires model checkpoints, software versions, preprocessing scripts, and the source BMPs. This is distinct from verifying the existing 405-row data table.

**Audit conclusion:** Full 405-row tabular coverage, the nine illustrative image identities, and the rank correlations are verified. Original fitted PLCC/RMSE provenance and raw-image generation remain separate documentation tasks.
