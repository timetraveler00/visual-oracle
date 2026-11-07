"""Read-only EXP01--EXP04 research progress discovery for a local Streamlit page.

The 2026-11-04 historical baseline is labelled 'reported'; no historic result is
silently treated as a newly verified run. EXP04 counts are based on on-disk jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

VIDEOS = {
    "Shields": {"file": "720p50_shields_ter.y4m", "role": "Development", "smoke": "job_exp04_smoke_shields"},
    "FourPeople": {"file": "720p50_fourpeople_exp04.y4m", "role": "Development", "smoke": "job_exp04_smoke_fourpeople"},
    "Parkrun": {"file": "720p50_parkrun_exp04.y4m", "role": "Holdout", "smoke": None},
    "ScreenShare": {"file": "720p50_screenshare_exp04.y4m", "role": "Holdout", "smoke": None},
}
FAULTS = ("blur", "color", "freeze", "black", "occlusion")
CONDITIONS = [("baseline", "medium")] + [(f, s) for f in FAULTS for s in ("low", "medium")]
PILOT_JOB_NAMES = [f"job_exp04_pilot_shields_b{i:02d}" for i in range(1, 4)]
NITS_RESULTS = [
    {"Metric": "OpenCLIP", "SROCC": 0.7297, "LOGO Accuracy (%)": 79.51},
    {"Metric": "DINOv2-Base", "SROCC": 0.7076, "LOGO Accuracy (%)": 77.53},
    {"Metric": "DINOv2-Large", "SROCC": 0.6765, "LOGO Accuracy (%)": 76.79},
    {"Metric": "DISTS", "SROCC": 0.7030, "LOGO Accuracy (%)": 81.98},
    {"Metric": "SSIM", "SROCC": 0.6565, "LOGO Accuracy (%)": 73.09},
]
HISTORICAL = [
    {"ID": "N01", "Study": "NITS-IQA severe distortions", "Scope": "9 pairs / 5 metrics", "Reported sessions": "—", "Status": "Completed (historical record)", "Remaining": "Expand results in manuscript"},
    {"ID": "N02", "Study": "NITS-IQA full dataset", "Scope": "405 pairs / MOS / 5 metrics", "Reported sessions": "—", "Status": "Completed (historical record)", "Remaining": "Verify final correlation tables"},
    {"ID": "N03", "Study": "MOS>=50 LOGO classification", "Scope": "405 pairs / grouped validation", "Reported sessions": "—", "Status": "Completed (historical record)", "Remaining": "Keep separate from software defect detection"},
    {"ID": "N04", "Study": "Threshold sensitivity", "Scope": "5 metrics / threshold shifts", "Reported sessions": "—", "Status": "Completed (historical record)", "Remaining": "Calibrate Jitsi thresholds separately"},
    {"ID": "G01", "Study": "Google Meet Selenium/Grid", "Scope": "Live integration", "Reported sessions": "Not consolidated", "Status": "Completed (historical record)", "Remaining": "Independent fault validation"},
    {"ID": "EXP01", "Study": "Blur reproducibility", "Scope": "Shields / 4 conditions", "Reported sessions": 20, "Status": "Completed (historical record)", "Remaining": "Cross-content generalization"},
    {"ID": "EXP02", "Study": "Multiple source-side faults", "Scope": "Shields / 6 conditions", "Reported sessions": 6, "Status": "Completed (historical record)", "Remaining": "Independent repetitions"},
    {"ID": "EXP03", "Study": "Evaluation-scale sensitivity", "Scope": "Shields / baseline+blur", "Reported sessions": 10, "Status": "Completed (historical record)", "Remaining": "320x180 secondary analysis"},
]
REVIEW_GAPS = [
    ("Nine-pair sample too small", "NITS-IQA 405-pair analysis", "Integrate full analysis in revised paper", "High"),
    ("Correlation / sensitivity", "SROCC, LOGO, sensitivity results", "Check PLCC and complete reporting", "High"),
    ("Layered oracle not validated", "Protocol / candidate policy", "Paired single vs flat vs layered held-out comparison", "Critical"),
    ("Thresholds engineered", "NITS-IQA calibration separate", "Calibrate on Jitsi development set; freeze", "Critical"),
    ("FPR / FNR on actual faults", "EXP02 exploratory only", "Independent EXP04 labeled session/episode evaluation", "Critical"),
    ("Resolution sensitivity", "EXP03 + smoke 640x360", "Mask/ROI audit and 320x180 secondary study", "Medium"),
    ("Runtime 50-80 ms", "Earlier pipeline observation", "Measure per-metric and end-to-end latency", "High"),
    ("Generalization", "Four videos prepared", "Locked Parkrun/ScreenShare evaluation", "Critical"),
    ("Network-induced freeze", "Only source-side freeze so far", "Separate network fault experiment if claimed", "Secondary"),
]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        return default


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError, UnicodeError):
        return []


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_status(job_dir: Path) -> dict:
    # status_recovery.json may be newer after a transient Windows file-lock error.
    candidates = [p for p in (job_dir / "status.json", job_dir / "status_recovery.json") if p.is_file()]
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        status = read_json(path)
        if isinstance(status, dict):
            return status
    return {}


def job_record(job_dir: Path) -> dict:
    cfg = read_json(job_dir / "scenario.json", {}) or {}
    state = get_status(job_dir)
    conditions = cfg.get("conditions") if isinstance(cfg.get("conditions"), list) else []
    planned = len(conditions) * max(1, to_int(cfg.get("repeats"), 1))
    results = state.get("results") if isinstance(state.get("results"), list) else []
    valid_results = [r for r in results if isinstance(r, dict) and r.get("run_valid") is not False]
    invalid_results = [r for r in results if isinstance(r, dict) and r.get("run_valid") is False]
    status = str(state.get("state") or ("configured" if cfg else "unknown"))
    completed = bool(status == "completed" and planned > 0 and len(results) == planned and not invalid_results)
    return {
        "job": job_dir.name,
        "video": str(cfg.get("video", "")),
        "phase": str(cfg.get("exp04_phase", "")),
        "status": status,
        "planned": planned,
        "recorded": len(results),
        "valid": len(valid_results),
        "invalid": len(invalid_results),
        "complete": completed,
        "updated_at": str(state.get("updated_at", "")),
        "results": results,
        "config": cfg,
    }


def discover_jobs(root: Path) -> dict[str, dict]:
    jobs_dir = root / "captures" / "dashboard_jobs"
    if not jobs_dir.is_dir():
        return {}
    return {p.name: job_record(p) for p in sorted(jobs_dir.glob("job_exp04_*")) if p.is_dir()}


def smoke_csv_path(root: Path) -> Path | None:
    candidates = [root / "EXP04_smoke_640_runs.csv"]
    analysis_dir = root / "captures" / "exp04_smoke_640"
    if analysis_dir.is_dir():
        candidates.extend(analysis_dir.glob("*/EXP04_smoke_640_runs.csv"))
    candidates = [p for p in candidates if p.is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None



def fault_log_path(root: Path) -> Path | None:
    """Prefer the live aggregate emitted by fault injection; never merge snapshots."""
    candidates = [root / "captures" / "fault_runs_summary.csv",
                  root / "fault_runs_summary.csv"]
    return next((p for p in candidates if p.is_file()), None)


def read_fault_log(root: Path) -> tuple[list[dict], str | None]:
    path = fault_log_path(root)
    if not path:
        return [], None
    rows = read_csv(path)
    required = {"scenario", "severity", "run_folder", "sample_count", "quality_ok",
                "mean_ssim_all"}
    if rows and not required.issubset(rows[0]):
        return [], f"{path} (unsupported columns)"
    # De-duplicate only exact capture-folder matches, preserving the latest entry.
    by_folder = {}
    for row in rows:
        folder = row.get("run_folder", "").strip().replace("/", "\\").lower()
        if folder:
            by_folder[folder] = row
    return list(by_folder.values()), str(path)


def fault_log_statistics(rows: list[dict]) -> dict:
    samples = sum(to_int(r.get("sample_count")) for r in rows)
    quality_ok = sum(to_int(r.get("quality_ok")) for r in rows)
    by_condition = {}
    for r in rows:
        key = (r.get("scenario", ""), r.get("severity", ""))
        by_condition[key] = by_condition.get(key, 0) + 1
    return {"runs": len(rows), "samples": samples, "quality_ok": quality_ok,
            "quality_pct": round(100 * quality_ok / samples, 2) if samples else 0,
            "by_condition": by_condition}


def normalize_condition(row: dict) -> str:
    scenario = str(row.get("scenario", "")).lower().strip()
    severity = str(row.get("severity", "")).lower().strip()
    if scenario == "baseline" and severity in ("none", "medium"):
        severity = "medium"
    return f"{scenario}/{severity}"


def csv_pilot_candidates(root: Path, rows: list[dict]) -> dict:
    """Evidence only: match randomized job order, never infer run_valid from CSV.

    A job status is authoritative for validity; matching the manifest's exact
    11-run randomized order is strong supporting evidence but not a job result.
    """
    manifest = read_json(root / "captures" / "dashboard_jobs" /
                         "EXP04_pilot_shields_plan.json", {}) or {}
    blocks = manifest.get("jobs", []) if isinstance(manifest, dict) else []
    if not isinstance(blocks, list):
        blocks = []
    used = set()
    matches = []
    for block in blocks:
        expected = block.get("order", []) if isinstance(block, dict) else []
        name = str(block.get("name", "")) if isinstance(block, dict) else ""
        if name not in PILOT_JOB_NAMES or len(expected) != 11:
            continue
        possibilities = []
        for start in range(len(rows) - len(expected) + 1):
            indices = list(range(start, start + len(expected)))
            if any(i in used for i in indices):
                continue
            window = [rows[i] for i in indices]
            if [normalize_condition(r) for r in window] != expected:
                continue
            # Keep old EXP01--03 and technical smoke records out of pilot counts.
            if any("joint_exp04_" not in str(r.get("run_folder", "")).lower()
                   for r in window):
                continue
            if any(to_int(r.get("sample_count")) < 25 for r in window):
                continue
            # Require monotonically increasing timestamped capture folders.
            stamps = [re.search(r"20\d{6}_\d{6}", str(r.get("run_folder", "")))
                      for r in window]
            if any(m is None for m in stamps):
                continue
            stamps = [m.group(0) for m in stamps]
            if stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
                continue
            possibilities.append((start, indices, window, stamps))
        if possibilities:
            _, indices, window, stamps = possibilities[-1]
            used.update(indices)
            matches.append({"job": name, "rows": window,
                            "runs": len(window), "samples": sum(to_int(r.get("sample_count")) for r in window),
                            "quality_ok": sum(to_int(r.get("quality_ok")) for r in window),
                            "first_timestamp": stamps[0], "last_timestamp": stamps[-1]})
    return {"matches": matches, "runs": sum(m["runs"] for m in matches),
            "samples": sum(m["samples"] for m in matches),
            "quality_ok": sum(m["quality_ok"] for m in matches),
            "matched_rows": [r for m in matches for r in m["rows"]]}


def artifact_found(root: Path, name: str) -> bool:
    candidates = [root / name, root / "captures" / name]
    for base in (root / "analysis", root / "captures" / "exp01_analysis",
                 root / "captures" / "exp02_analysis", root / "captures" / "exp03_analysis",
                 root / "EXP03_Resolution_Study"):
        candidates.append(base / name)
    return any(p.is_file() for p in candidates)


def main_jobs(jobs: dict[str, dict]) -> dict[str, list[dict]]:
    """Only jobs explicitly marked/named EXP04 main, never smoke or pilot."""
    output = {v: [] for v in VIDEOS}
    for job in jobs.values():
        name = job["job"].lower()
        phase = job["phase"].lower()
        if not (phase in ("main", "development_main", "holdout_main", "validation") or
                name.startswith(("job_exp04_main_", "job_exp04_dev_", "job_exp04_holdout_"))):
            continue
        source = Path(job["video"]).name.lower()
        for video, spec in VIDEOS.items():
            if source == spec["file"].lower():
                output[video].append(job)
                break
    return output


def condition_progress(jobs: dict[str, dict]) -> list[dict]:
    counts = {(c, s): {"valid": 0, "invalid": 0} for c, s in CONDITIONS}
    for name in PILOT_JOB_NAMES:
        job = jobs.get(name)
        if not job:
            continue
        for result in job["results"]:
            if not isinstance(result, dict):
                continue
            key = (str(result.get("scenario", "")), str(result.get("severity", "medium")))
            if key in counts:
                bucket = "invalid" if result.get("run_valid") is False else "valid"
                counts[key][bucket] += 1
    return [{"Condition": "baseline" if c == "baseline" else f"{c}/{s}",
             "Valid": counts[(c, s)]["valid"], "Invalid": counts[(c, s)]["invalid"],
             "Target": 3, "Remaining": max(0, 3 - counts[(c, s)]["valid"])}
            for c, s in CONDITIONS]


def snapshot(root: Path) -> dict:
    root = root.resolve()
    jobs = discover_jobs(root)
    pilot = [jobs[name] for name in PILOT_JOB_NAMES if name in jobs]
    pilot_valid = sum(j["valid"] for j in pilot)
    pilot_invalid = sum(j["invalid"] for j in pilot)
    main = main_jobs(jobs)
    main_valid = {name: sum(j["valid"] for j in runs) for name, runs in main.items()}
    log_rows, log_path = read_fault_log(root)
    log_stats = fault_log_statistics(log_rows)
    pilot_csv = csv_pilot_candidates(root, log_rows)
    smoke_path = smoke_csv_path(root)
    smoke_rows = read_csv(smoke_path) if smoke_path else []
    smoke_jobs = {}
    for name, info in VIDEOS.items():
        smoke_job_name = info["smoke"]
        job = jobs.get(smoke_job_name) if smoke_job_name else None
        csv_runs = [r for r in smoke_rows if r.get("job") == smoke_job_name]
        # If only CSV exists, label it as CSV evidence rather than a verified job status.
        smoke_jobs[name] = {"job": job, "csv_runs": len(csv_runs),
                            "confirmed_runs": job["valid"] if job else 0}
    files = {name: (root / "videos" / info["file"]).is_file() for name, info in VIDEOS.items()}
    artifacts = {
        "EXP01": artifact_found(root, "EXP01_analysis_report.md"),
        "EXP02": artifact_found(root, "EXP02_pilot_analysis_report.md"),
        "EXP03": artifact_found(root, "EXP03_report.md"),
        "thresholds_frozen": artifact_found(root, "EXP04_thresholds_frozen.json"),
        "session_predictions": artifact_found(root, "EXP04_session_predictions.csv"),
        "method_comparison": artifact_found(root, "EXP04_method_comparison.csv"),
        "latency": artifact_found(root, "EXP04_latency.csv"),
    }
    return {"root": str(root), "jobs": jobs, "pilot": pilot,
            "pilot_valid": pilot_valid, "pilot_invalid": pilot_invalid,
            "pilot_csv": pilot_csv,
            "fault_log_rows": log_rows, "fault_log_path": log_path,
            "fault_log_stats": log_stats,
            "main": main, "main_valid": main_valid,
            "main_total": sum(main_valid.values()), "smoke": smoke_jobs,
            "smoke_rows": smoke_rows, "smoke_path": str(smoke_path) if smoke_path else None,
            "video_files": files, "artifacts": artifacts,
            "conditions": condition_progress(jobs),
            "pilot_plan_exists": (root / "captures" / "dashboard_jobs" / "EXP04_pilot_shields_plan.json").is_file(),
            "historical_reported_jitsi": 36, "historical_reported_smoke": 6,
            "target_pilot": 33, "target_main": 240}


def matrix_rows(s: dict) -> list[dict]:
    rows = list(HISTORICAL)
    video_count = sum(s["video_files"].values())
    smoke_completed = sum(min(3, x["confirmed_runs"]) for x in s["smoke"].values())
    smoke_csv_runs = sum(x["csv_runs"] for x in s["smoke"].values())
    pilot_valid = s["pilot_valid"]
    main_valid = s["main_total"]
    rows.extend([
        {"ID": "EXP04-A", "Study": "Prepare four source videos", "Scope": f"{video_count}/4 source files present", "Reported sessions": "—", "Status": "Complete (file check)" if video_count == 4 else "Incomplete", "Remaining": "Verify source hashes / provenance" if video_count == 4 else "Prepare missing source videos"},
        {"ID": "EXP04-B", "Study": "Shields + FourPeople smoke", "Scope": f"{smoke_completed}/6 job-verified; {smoke_csv_runs}/6 CSV", "Reported sessions": 6, "Status": "Complete (job logs)" if smoke_completed == 6 else "Reported / CSV" if smoke_csv_runs == 6 else "Incomplete", "Remaining": "Other fault types belong to pilot"},
        {"ID": "EXP04-C", "Study": "640x360 smoke re-analysis", "Scope": f"{len(s['smoke_rows'])}/6 run summaries", "Reported sessions": "Same 6 smoke runs", "Status": "Complete (CSV)" if len(s["smoke_rows"]) == 6 else "Not found", "Remaining": "Full-metric scoring and ROI audit"},
        {"ID": "EXP04-D", "Study": "Shields pilot", "Scope": f"{pilot_valid}/33 job-valid; {s['pilot_csv']['runs']}/33 CSV-order evidence; {s['pilot_invalid']} invalid", "Reported sessions": 33, "Status": "Complete" if pilot_valid >= 33 else "In progress" if pilot_valid else "Configured" if s["pilot_plan_exists"] else "Not started", "Remaining": f"{max(0,33-pilot_valid)} valid independent sessions"},
        {"ID": "EXP04-E", "Study": "Main Jitsi experiment", "Scope": f"{main_valid}/240 valid runs", "Reported sessions": 240, "Status": "Complete" if main_valid >= 240 else "In progress" if main_valid else "Not started", "Remaining": f"{max(0,240-main_valid)} valid independent sessions"},
        {"ID": "EXP04-F", "Study": "Single / flat / layered comparison", "Scope": "Session and episode predictions", "Reported sessions": "—", "Status": "Artifacts found (review)" if s["artifacts"]["method_comparison"] else "Not started", "Remaining": "Balanced accuracy, F1, FPR/FNR, confidence intervals"},
        {"ID": "EXP04-G", "Study": "Latency and runtime", "Scope": "Per-metric and end-to-end", "Reported sessions": "—", "Status": "Artifacts found (review)" if s["artifacts"]["latency"] else "Not started", "Remaining": "Measured latency and p95"},
    ])
    return rows


def review_rows(s: dict) -> list[dict]:
    rows = []
    for issue, evidence, remaining, priority in REVIEW_GAPS:
        rows.append({"Reviewer issue": issue, "Current evidence": evidence,
                     "Remaining action": remaining, "Priority": priority})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only EXP04 research dashboard data check")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    s = snapshot(args.root)
    print("Project:", s["root"])
    print("Source videos:", sum(s["video_files"].values()), "/ 4")
    print("EXP04 smoke job-verified:", sum(min(3, x["confirmed_runs"]) for x in s["smoke"].values()), "/ 6")
    print("EXP04 smoke CSV:", len(s["smoke_rows"]), "/ 6")
    print("EXP04 pilot valid:", s["pilot_valid"], "/ 33; invalid:", s["pilot_invalid"])
    print("EXP04 main valid:", s["main_total"], "/ 240")
    print("Fault log:", s["fault_log_path"] or "not found")
    print("Fault log rows:", s["fault_log_stats"]["runs"],
          "quality:", f"{s['fault_log_stats']['quality_ok']}/{s['fault_log_stats']['samples']}")
    print("Pilot CSV order evidence:", s["pilot_csv"]["runs"], "/ 33",
          "(not a replacement for job validity)")
    for match in s["pilot_csv"]["matches"]:
        print("Pilot CSV", match["job"], match["runs"], "runs",
              f"{match['quality_ok']}/{match['samples']} quality samples")
    print("Smoke CSV:", s["smoke_path"] or "not found")
    for job in s["pilot"]:
        print("Pilot", job["job"], job["status"], f"{job['valid']}/{job['planned']}")


if __name__ == "__main__":
    main()
