# Visual Oracle: publication snapshot (EXP04)


## Associated manuscript

**Title:** *Automated Visual Test Oracles for Video-Conferencing Software: An Empirical Evaluation of Semantic and Structural Image Similarity Metrics*

**Authors (in manuscript order):** Doğukan Biçer; Savaş Öztürk.

**Publication status:** Manuscript under peer review/revision as of 2026-11-07.
This repository does not claim journal acceptance, publication, or a DOI.
See [`CITATION.cff`](CITATION.cff) for a citation of this repository snapshot;
the manuscript and this repository are distinct citation objects.

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
The earlier Google Meet automation prototype was abandoned because of
authentication constraints. Its obsolete setup notes and example configurations
are excluded from this Jitsi-focused publication snapshot. Historical Google
Meet study labels are retained where needed to document experimental provenance.

## Snapshot status

- Published files from source: 352 (some are English adaptations).
- Excluded/review entries in the local scan: 24694.
- The source directory was not modified.
- See `publication_manifest.csv` for original and published SHA-256 values.
  Rows marked `english_adaptation` are not byte-identical to the local originals.
  The frozen EXP04 scorer, plan and calibration lock are unchanged.

## Verification and inspection

The read-only verifier in this repository checks the Git commit's stored bytes
against `publication_manifest.csv`, validates the 240 planned scenarios and
frozen plan/scorer hashes, and tests the four nested result archives. It uses
only Python's standard library and the Git command-line client. From the
repository root, run (on Windows, use Anaconda Prompt if `python` is not on
PowerShell's PATH):

```bash
python tools/verify_git_manifest.py --repo . --ref HEAD
```

A successful run reports `352 / 352` manifest matches, `240 / 240` scenario
matches, four verified ZIPs, and `[PASS]`. This checks integrity, not scientific
reproduction from omitted raw videos, images, or model checkpoints.

The EXP04 holdout tables are stored **inside**
`captures/EXP04_holdout_evaluation_20261107_071300.zip`. For example, to
extract them into a separate folder for inspection:

```bash
python -m zipfile -e captures/EXP04_holdout_evaluation_20261107_071300.zip extracted_holdout
```

The resulting `extracted_holdout/EXP04_holdout_5metric_sessions.csv` is the
actual session-level table. Historical scripts may refer to paths not shipped
in this snapshot; those paths are not prerequisites for integrity verification.

## Licensing and third-party materials

The authors' **original source code and accompanying software documentation**
are released under the [MIT License](LICENSE), subject to confirmation that
the named copyright holders have the rights to license those contributions.
See [LICENSING.md](LICENSING.md) for the precise scope and outstanding rights
checks.

**Research data and third-party materials are not covered by the MIT License.**
This includes recorded and derived result tables, experiment metadata and ZIP
archives, external test videos, the NITS-IQA benchmark, downloaded model
weights, and third-party software. No separate open-data license has yet been
assigned to the published derived data. The absence of a data license should
not be interpreted as unrestricted permission to redistribute those files.
