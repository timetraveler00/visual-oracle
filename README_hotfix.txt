Visual Oracle v0.2/v0.3 — Windows status & stalled receiver hotfix

1. Stop the Streamlit server with Ctrl+C. Ensure no experiment job is running.
2. Extract fix_visual_oracle_status_and_stall.py into D:\visual-oracle-revision.
3. In Anaconda Prompt:
   conda activate visual-oracle
   cd /d D:\visual-oracle-revision
   python fix_visual_oracle_status_and_stall.py
   python -m streamlit run 12_visual_oracle_dashboard.py --server.address 127.0.0.1 --server.port 8501

The patch modifies three files and backs each up with .backup_YYYYMMDD_HHMMSS:
  12_visual_oracle_runner.py
  10_jitsi_joint_quality_performance.py
  12_visual_oracle_dashboard.py

Changes:
- Retries Windows status.json replacements and uses status_recovery.json as a fallback.
- Excludes PNG captures from quality averages if the preceding WebRTC interval decoded zero new frames and received zero video bytes.
- Flags experiment runs with >=2 consecutive stalled intervals as INVALID in the dashboard.
- Excludes flagged runs from new condition comparison charts; keeps raw data for diagnosis.

This is not a network repair. It protects research validity and avoids false job failures.
Prior runs are NOT reclassified; inspect previous logs and exclude stalled sessions manually.
After installing a fresh v0.3 package, run the hotfix again if it overwrote the patched files.

No Selenium Grid, Jitsi, Docker, or existing captured files are changed.
The hotfix is syntax-checked against both v0.2 and v0.3, not live-tested on Windows.
