# Visual Oracle — v0.4 Measurement Validation Hotfix

## Why this patch exists

In previous runs, Jitsi appeared to switch its active video/peer-connection around 25–30 seconds. The old stats selector preferred a previously selected `inbound-rtp` record even when its counters stopped, while the HTML video playback counter and received frame barcodes continued to advance. An earlier hotfix incorrectly marked such samples as `STALE` on the basis of RTP stats alone.

This patch **does not claim to prove that the root cause is a switched RTP record**. It adds enough diagnostics to test that hypothesis and prevents the earlier false-positive rule from discarding playing video frames.

## Compatibility

- Works with Visual Oracle dashboard v0.2 and v0.3, with or without the earlier status/stall hotfix.
- Updates `09_jitsi_webrtc_stats.py`, `10_jitsi_joint_quality_performance.py`, `12_visual_oracle_runner.py`, `12_visual_oracle_dashboard.py`.
- Saves timestamped backups of every modified Python file.
- Does not delete videos, scenarios, Docker containers or results.
- Tested: syntax and patch installation against v0.2, v0.3 and v0.2 + previous hotfix; synthetic active-stream selection regression test.
- Not yet tested end-to-end on the user's Windows/Docker/Jitsi environment.

## Install in Windows Anaconda Prompt

1. Stop Streamlit with `Ctrl+C` and ensure no experiment is running.
2. Extract `fix_webrtc_tracking_v04.py` and `validate_baseline_v04.py` to `D:\visual-oracle-revision`.
3. Run:

```bat
conda activate visual-oracle
cd /d D:\visual-oracle-revision
python fix_webrtc_tracking_v04.py
python -m streamlit run 12_visual_oracle_dashboard.py --server.address 127.0.0.1 --server.port 8501
```

## First run — baseline only

- Video: an original source file under `videos/` (use the same video in subsequent tests).
- Nodes: 2.
- Conditions: **Baseline only**.
- Duration: **60 seconds** (necessary to observe the previously reported 30-second transition).
- Sample interval: **5 seconds**.
- Repeats: 1.
- Perceptual metrics: **none for the first validation** (isolate transport/statistics).

After completion, stop Streamlit if necessary, then run:

```bat
python validate_baseline_v04.py
```

Expected new columns in `captures\joint_...\joint_measurements.csv`:

- `inbound_count`, `pc_index`, `stats_stream_changed`
- `video_dom_uid`, `video_track_ids`
- `playback_delta_frames`, `observed_playback_fps`
- `stream_liveness` (`LIVE`, `UNKNOWN`, `STALLED`)
- `stats_selection_note` (including `RTP_STATS_MISMATCH_PLAYBACK_ACTIVE`)

Expected new summary fields: `mean_observed_playback_fps`, `liveness_counts`, `stats_stream_switches`, `rtp_playback_mismatch_samples`.

**Important:** `LIVE` is an observation from WebRTC or playback counters; it is not a substitute for frame-barcode synchronization or a formal validity study. If the first test is successful, next run Baseline + Blur Low (60 seconds, 5-second interval, 1 repeat each) and compare both metrics and playback liveness. Only then expand the fault matrix.

## Recovery

To undo the patch, stop Streamlit and copy each `*.backup_YYYYMMDD_HHMMSS` file back over its original `.py` file. Backups are deliberately not removed.
