"""EXP01--EXP04 research tracking page for the existing Streamlit dashboard.

Install in ROOT/pages/; no edits to 12_visual_oracle_dashboard.py needed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exp04_research_monitor import (  # noqa: E402
    VIDEOS, NITS_RESULTS, PILOT_JOB_NAMES, snapshot, matrix_rows, review_rows, read_json,
)

st.set_page_config(page_title="Research Matrix | Visual Oracle", page_icon="📊", layout="wide")

st.title("Visual Oracle | Research Matrix")
st.caption("EXP01–EXP04 · NITS-IQA · Google Meet/Jitsi · Applied Sciences revision")
st.info("The dashboard reads local job files. Historical experiment counts are labelled as previously reported; "
        "only EXP04 job records are counted as newly observed sessions. Source-side freeze is not a network freeze.")


def show_progress(label: str, count: int, total: int, caption: str = ""):
    st.write(f"**{label}** — {count}/{total}")
    st.progress(min(1.0, max(0.0, count / total)), text=caption or f"{max(0, total - count)} remaining")


def render_live():
    s = snapshot(ROOT)
    tab_overview, tab_matrix, tab_live, tab_fault, tab_results, tab_review = st.tabs([
        "Overview", "All experiments", "EXP04 live monitor", "Fault log (CSV)", "Results", "Reviewer gaps"
    ])
    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prior Jitsi sessions (reported)", "36", help="EXP01=20, EXP02=6, EXP03=10. Not new EXP04 sessions.")
        c2.metric("EXP04 smoke (reported)", "6", help="Shields 3 + FourPeople 3; technical checks only.")
        c3.metric("Pilot valid", f"{s['pilot_valid']}/33")
        c4.metric("Main valid", f"{s['main_total']}/240")
        if s["pilot_csv"]["runs"]:
            st.info(f"Pilot CSV evidence: {s['pilot_csv']['runs']}/33 runs match the randomized block order. "
                    f"Quality samples: {s['pilot_csv']['quality_ok']}/{s['pilot_csv']['samples']}. "
                    "Job status determines validity; CSV-only evidence is not counted as valid runs.")
        st.caption("The 640x360 re-analysis does not add independent Jitsi sessions. "
                   "Historical 36 + smoke 6 = 42 previously reported sessions; new pilot/main counts are separate.")
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("EXP04 progress")
            show_progress("Source videos", sum(s["video_files"].values()), 4)
            smoke_verified = sum(min(3, x["confirmed_runs"]) for x in s["smoke"].values())
            show_progress("Smoke job logs", smoke_verified, 6, "6 smoke runs are reported; the CSV is shown separately")
            show_progress("Shields pilot (job-verified)", s["pilot_valid"], 33,
                          f"{s['pilot_invalid']} invalid; only job-verified runs counted")
            if s["pilot_csv"]["runs"]:
                show_progress("Pilot measured (CSV-order evidence)", s["pilot_csv"]["runs"], 33,
                              "Not yet verified against job status.json")
            show_progress("Main development + holdout", s["main_total"], 240)
        with right:
            st.subheader("Evidence checks")
            checks = [
                ("4 video files", all(s["video_files"].values())),
                ("640x360 smoke CSV (6 rows)", len(s["smoke_rows"]) == 6),
                ("Pilot plan JSON", s["pilot_plan_exists"]),
                ("Threshold freeze artifact", s["artifacts"]["thresholds_frozen"]),
                ("Method comparison CSV", s["artifacts"]["method_comparison"]),
                ("Latency CSV", s["artifacts"]["latency"]),
            ]
            for label, present in checks:
                st.write(("✅" if present else "⬜") + " " + label)
            if s["pilot_invalid"]:
                st.warning(f"Pilot has {s['pilot_invalid']} run(s) marked invalid. Do not count them as successful sessions.")
        st.subheader("Next research milestones")
        if s["pilot_valid"] < 33:
            st.markdown("**1. Shields pilot:** Complete 11 conditions × 3 independent sessions, starting with block b01.")
        elif s["main_total"] < 240:
            st.markdown("**1. Calibrate and freeze policies**, then execute development and locked holdout sessions.")
        else:
            st.markdown("**1. Analyze locked test predictions**, session/episode accuracy and inference latency.")
        st.markdown("**2. Scientific controls:** fixed 640×360 ROI/masking, no upscaling, temporal freeze logic, "
                    "missing-data policy, threshold freeze and no holdout leakage.")
        st.markdown("**3. Final comparison:** single vs TC11 vs flat fusion vs layered, using the same evidence pool.")
    with tab_matrix:
        st.subheader("All experiments and missing work")
        df = pd.DataFrame(matrix_rows(s))
        st.dataframe(df, hide_index=True, use_container_width=True, height=610)
        st.download_button("Download experiment matrix (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="EXP04_research_matrix.csv", mime="text/csv")
        st.caption("Completed historical rows are based on earlier experiment reports; "
                   "EXP04-D/E counts are read from local status.json/status_recovery.json, not from planned scenario.json.")
    with tab_live:
        st.subheader("Video × phase matrix")
        video_rows = []
        for video, info in VIDEOS.items():
            smoke = s["smoke"][video]
            main_valid = s["main_valid"][video]
            video_rows.append({"Video": video, "Role": info["role"],
                               "Source exists": "Yes" if s["video_files"][video] else "No",
                               "Smoke (job logs)": f"{smoke['confirmed_runs']}/3" if info["smoke"] else "Locked — not run",
                               "Pilot": f"{s['pilot_valid']}/33" if video == "Shields" else "—",
                               "Main": f"{main_valid}/60"})
        st.dataframe(pd.DataFrame(video_rows), hide_index=True, use_container_width=True)
        st.caption("Parkrun and ScreenShare remain held out until thresholds are frozen. "
                   "Smoke runs are excluded from the 33-session pilot and 240-session main experiment.")
        st.subheader("Shields pilot: condition coverage")
        condition_df = pd.DataFrame(s["conditions"])
        st.dataframe(condition_df, hide_index=True, use_container_width=True)
        st.bar_chart(condition_df.set_index("Condition")[["Valid", "Remaining"]], height=290)
        if s["pilot_csv"]["matches"]:
            st.subheader("Pilot block CSV evidence (separate from job validity)")
            st.dataframe(pd.DataFrame([
                {"Block": m["job"], "CSV runs": m["runs"], "Quality samples": f"{m['quality_ok']}/{m['samples']}",
                 "First capture": m["first_timestamp"], "Last capture": m["last_timestamp"]}
                for m in s["pilot_csv"]["matches"]]), hide_index=True, use_container_width=True)
        st.subheader("EXP04 job files")
        if s["jobs"]:
            job_rows = [{"Job": j["job"], "State (last recorded)": j["status"],
                         "Recorded": j["recorded"], "Valid": j["valid"],
                         "Invalid": j["invalid"], "Planned": j["planned"],
                         "Updated": j["updated_at"]} for j in s["jobs"].values()]
            st.dataframe(pd.DataFrame(job_rows), hide_index=True, use_container_width=True)
        else:
            st.warning("No EXP04 job folders found in captures/dashboard_jobs.")
        st.caption("A configured job is not a completed session. 'Running' is the last saved state; "
                   "it does not prove the process is still alive. Invalid sessions are shown separately.")
    with tab_fault:
        st.subheader("Live fault-runs aggregate (diagnostic; native capture resolution)")
        stats = s["fault_log_stats"]
        if not s["fault_log_rows"]:
            st.warning("No captures/fault_runs_summary.csv found or its columns are unsupported.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique capture folders", stats["runs"])
            c2.metric("Quality-OK samples", f"{stats['quality_ok']}/{stats['samples']}")
            c3.metric("Sample quality", f"{stats['quality_pct']:.2f}%")
            st.caption("This cumulative log includes historical runs and smoke tests. "
                       "Do not add its total to EXP01–04 session counts. "
                       f"Source: {s['fault_log_path']}")
            df = pd.DataFrame(s["fault_log_rows"])
            df["Condition"] = df["scenario"].astype(str) + "/" + df["severity"].astype(str)
            df["Capture"] = df["run_folder"].astype(str).str.replace("/", "\\", regex=False).str.rsplit("\\", n=1).str[-1]
            for col in ("mean_ssim_all", "mean_ssim_fault_active", "mean_ssim_fault_inactive",
                        "mean_delta_e76_fault_active", "sample_count", "quality_ok"):
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            st.subheader("Capture history")
            st.dataframe(df[["Capture", "Condition", "quality_ok", "sample_count", "mean_ssim_all",
                             "mean_ssim_fault_active", "mean_ssim_fault_inactive",
                             "mean_delta_e76_fault_active"]], hide_index=True,
                         use_container_width=True, height=360)
            st.download_button("Download fault-log table", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="EXP04_fault_log_export.csv", mime="text/csv")
            if s["pilot_csv"]["matched_rows"]:
                st.subheader("Pilot blocks — native-resolution diagnostic metrics")
                pilot_df = pd.DataFrame(s["pilot_csv"]["matched_rows"])
                pilot_df["Condition"] = pilot_df["scenario"].astype(str) + "/" + pilot_df["severity"].astype(str)
                pilot_df["SSIM active"] = pd.to_numeric(pilot_df["mean_ssim_fault_active"], errors="coerce")
                pilot_df["SSIM inactive"] = pd.to_numeric(pilot_df["mean_ssim_fault_inactive"], errors="coerce")
                pilot_df["DeltaE76 active"] = pd.to_numeric(pilot_df["mean_delta_e76_fault_active"], errors="coerce")
                grouped = pilot_df.groupby("Condition", as_index=True)[["SSIM active", "DeltaE76 active"]].mean()
                st.bar_chart(grouped[["SSIM active"]], y_label="SSIM (native resolution)")
                st.dataframe(pilot_df[["Condition", "sample_count", "quality_ok", "SSIM active",
                                       "SSIM inactive", "DeltaE76 active"]], hide_index=True,
                             use_container_width=True)
                st.caption("CSV order matches the pilot manifest; validity still requires status.json. "
                           "Native-resolution SSIM and DeltaE76 are NOT standardized 640×360 scores. "
                           "These are descriptive results, not defect-detection accuracy or F1.")
    with tab_results:
        st.subheader("EXP04 smoke · standardized 640×360")
        if s["smoke_rows"]:
            df = pd.DataFrame(s["smoke_rows"])
            numeric = ["ssim_all_640", "ssim_active_640", "ssim_inactive_640",
                       "coverage_pct", "valid_640", "total_samples", "psnr_all_db_640", "delta_e76_all_640"]
            for col in numeric:
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Video"] = df["video"].map(lambda v: "FourPeople" if "fourpeople" in str(v).lower() else "Shields")
            df["Condition"] = df["condition"].astype(str) + "/" + df["severity"].astype(str)
            chart = df.pivot_table(index="Condition", columns="Video", values="ssim_all_640", aggfunc="mean")
            st.bar_chart(chart, y_label="SSIM (640×360)", height=310)
            st.dataframe(df[["Video", "Condition", "valid_640", "total_samples", "coverage_pct",
                             "ssim_all_640", "ssim_active_640", "ssim_inactive_640", "psnr_all_db_640"]],
                         hide_index=True, use_container_width=True)
            st.caption("Smoke data validate the pipeline, not a held-out accuracy/F1 estimate. "
                       f"CSV source: {s['smoke_path']}")
        else:
            st.warning("EXP04_smoke_640_runs.csv not found. Copy it into the project root or run the smoke analysis script.")
        st.subheader("NITS-IQA · 405 pairs (previously reported)")
        nits = pd.DataFrame(NITS_RESULTS)
        st.dataframe(nits, hide_index=True, use_container_width=True)
        st.caption("SROCC: MOS ranking correlation; LOGO Accuracy: MOS>=50 classification. "
                   "Neither is Jitsi defect-detection accuracy.")
        st.subheader("Historical analysis files")
        hist = pd.DataFrame([{"Study": name, "Analysis report found locally": "Yes" if s["artifacts"][name] else "No / not in known paths"}
                             for name in ("EXP01", "EXP02", "EXP03")])
        st.dataframe(hist, hide_index=True, use_container_width=True)
    with tab_review:
        st.subheader("Reviewer requests → evidence → missing work")
        gaps = pd.DataFrame(review_rows(s))
        st.dataframe(gaps, hide_index=True, use_container_width=True, height=440)
        st.download_button("Download reviewer gap matrix (CSV)", gaps.to_csv(index=False).encode("utf-8-sig"),
                           file_name="EXP04_reviewer_gaps.csv", mime="text/csv")
        st.warning("A sequential AND policy is not more accurate than the equivalent flat AND/OR decision. "
                   "Any claimed layered advantage must be demonstrated with a different decision rule "
                   "or a measured reduction in runtime under equal inputs.")
    st.divider()
    st.caption(f"Project root: {ROOT} · Live job status: captures/dashboard_jobs/job_exp04_*/status.json · "
               "Refresh interval: 15 seconds (where Streamlit supports fragments)")


if hasattr(st, "fragment"):
    @st.fragment(run_every="15s")
    def live_page():
        render_live()
    live_page()
else:
    if st.button("Refresh data"):
        st.rerun()
    render_live()

# Notes are deliberately outside the auto-refresh fragment. They are not evidence of completed runs.
st.subheader("Research notes (optional)")
NOTES = ROOT / "captures" / "research_dashboard_notes.json"
notes = read_json(NOTES, {}) or {}
choices = ["EXP01", "EXP02", "EXP03", "EXP04-A", "EXP04-B", "EXP04-C", "EXP04-D", "EXP04-E", "EXP04-F", "EXP04-G", "Manuscript"]
selected = st.selectbox("Stage", choices)
with st.form("research_notes"):
    value = st.text_area("Note", value=str(notes.get(selected, {}).get("text", "")), height=100,
                         key=f"note_{selected}",
                         help="Notes do not mark a job complete or change experiment evidence.")
    save = st.form_submit_button("Save note")
if save:
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    notes[selected] = {"text": value, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    tmp = NOTES.with_name(NOTES.name + ".tmp")
    try:
        tmp.write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, NOTES)
        st.success("Note saved locally; experiment status is unchanged.")
    except OSError as exc:
        st.error(f"Could not save note: {exc}")
