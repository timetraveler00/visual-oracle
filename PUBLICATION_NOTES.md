# Visual Oracle: publication snapshot (EXP04)

This directory was generated as a **code-and-derived-results snapshot** from a
locally archived, immutable experimental project. It is NOT a complete raw-data
archive and must not be described as a self-contained reproduction package.

## What is included

- Source code, relevant configuration, selected derived CSV/JSON results.
- `captures/dashboard_jobs/EXP04_main240_plan.json` (240-session plan).
- `captures/dashboard_jobs/EXP04_main240_calibration_lock.json` (frozen thresholds).
- `exp04_oracle_frozen.py` (frozen decision logic).
- Available per-session `scenario.json` files and selected compact result ZIPs.
- `nits_iqa/verified/` (if present): 405-pair NITS-IQA scores, audit code,
  and integrity report. NITS-IQA is a separate benchmark from EXP04.
- `nits_iqa/analysis_notebooks/` and `nits_iqa/analysis_results/` (if present):
  English publication adaptations of seven historical notebooks and derived tables.
  The notebook code was validated against the 405-row tabular dataset; no source
  images or neural model inference are included. Historical originals remain local.
- `publication_manifest.csv` with per-file SHA-256 values.

## What is intentionally NOT included

- Raw Y4M/video input, captured PNG frames, fault-source videos, Docker caches.
- Downloaded model weights and large intermediate/diagnostic artifacts.
- Local-only `nits_iqa/legacy_original/` and `nits_iqa/source_workbook/`.
- Chrome/browser user profiles and actual runtime `config.json` files.
- Environment caches, logs, credentials, temporary or draft files.

The full experiment cannot be re-run from this repository alone. Before
publication, document the video licensing/source, checkpoint identifiers,
software environment, and how qualified researchers can access raw records
(e.g., a separately deposited archive with DOI). Do not claim unrestricted
public availability unless it has been established.

## Methodological note

The frozen EXP04 oracle was evaluated on independent Jitsi sessions. The
offline per-metric benchmark does not measure full end-to-end WebRTC latency.

The publication export passed an automated English-language marker audit. This
is not a substitute for a human review of ASCII-only prose and metadata.
Some historical Google Meet scripts relied on Turkish interface text and are
excluded rather than modified, because removing those strings changes behavior.

## Snapshot status

- Published files from source: 355 (some are English adaptations).
- Excluded/review entries in the local scan: 24694.
- The source directory was not modified.
- See `publication_manifest.csv` for original and published SHA-256 values.
  Rows marked `english_adaptation` are not byte-identical to the local originals.
  The frozen EXP04 scorer, plan and calibration lock are unchanged.

## Verification

Run `python visual_oracle_github_prepare_v4.py --verify` from the external
preparation script location, using `--output` if not the default.

## License

Choose and add a LICENSE file only after confirming rights to all code and
third-party components. Media and model weights are not relicensed here.
