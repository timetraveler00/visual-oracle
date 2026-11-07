"""Create three randomized EXP04 Shields pilot job configurations (11 runs each).

Usage (Windows Anaconda Prompt):
    cd /d D:\\visual-oracle-revision
    python exp04_create_shields_pilot.py

This script ONLY creates new scenario.json files and a plan manifest.
It does not launch Jitsi, change videos, overwrite existing jobs, or analyze data.
Run each job separately using 12_visual_oracle_runner.py (no --resume support).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

SOURCE_REL = "videos/720p50_shields_ter.y4m"
JOB_PREFIX = "job_exp04_pilot_shields_b"
DEFAULT_SEED = 20261104
FAULTS = ("blur", "color", "freeze", "black", "occlusion")
CONDITIONS = [{"scenario": "baseline", "severity": "medium"}] + [
    {"scenario": fault, "severity": severity}
    for fault in FAULTS for severity in ("low", "medium")
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def plan(root: Path, seed: int):
    source = root / SOURCE_REL
    if not source.is_file():
        raise FileNotFoundError(f"Missing source video: {source}")
    if not (root / "12_visual_oracle_runner.py").is_file():
        raise FileNotFoundError("Run from project root containing 12_visual_oracle_runner.py")
    jobs_root = root / "captures" / "dashboard_jobs"
    configs = []
    for block in range(1, 4):
        conditions = [dict(c) for c in CONDITIONS]
        random.Random(seed + block).shuffle(conditions)
        name = f"{JOB_PREFIX}{block:02d}"
        cfg = {
            "schema_version": 1,
            "name": f"EXP04 Shields randomized pilot block {block}/3 (11 independent Jitsi sessions)",
            "video": SOURCE_REL,
            "node_count": 2,
            "duration": 30,
            "interval": 1,
            "repeats": 1,
            "clip_seconds": 10,
            "perceptual_metrics": [],
            "conditions": conditions,
            "methodology": (
                "EXP04 DEVELOPMENT PILOT ONLY; 11 conditions randomized per block; "
                "3 blocks = 33 independent Jitsi sessions. "
                "Source-side faults; 640x360 scoring performed OFFLINE. "
                "No held-out Parkrun/ScreenShare data; no test threshold tuning. "
                "The runner's native-resolution metrics are diagnostic only."
            ),
            "exp04_block": block,
            "exp04_seed": seed + block,
        }
        configs.append((jobs_root / name / "scenario.json", cfg))
    counts = Counter((c["scenario"], c["severity"]) for _, cfg in configs for c in cfg["conditions"])
    assert len(counts) == 11 and set(counts.values()) == {3}, counts
    manifest = {
        "experiment": "EXP04", "phase": "development_pilot", "video": SOURCE_REL,
        "video_sha256": sha256(source), "base_seed": seed,
        "session_count": 33, "blocks": 3, "conditions_per_block": 11,
        "independent_unit": "one newly started Jitsi meeting per condition/run",
        "evaluation": "offline 640x360; no upscaling; same ROI/mask for all metrics",
        "note": "Smoke sessions and earlier EXP01-03 sessions are excluded from pilot counts.",
        "jobs": [
            {
                "name": p.parent.name,
                "config": str(p.relative_to(root)).replace("\\", "/"),
                "scenario_sha256": hashlib.sha256(render(cfg).encode("utf-8")).hexdigest(),
                "order": [f"{c['scenario']}/{c['severity']}" for c in cfg["conditions"]],
            }
            for p, cfg in configs
        ],
    }
    manifest_path = jobs_root / "EXP04_pilot_shields_plan.json"
    files = [(p, render(cfg)) for p, cfg in configs] + [(manifest_path, render(manifest))]
    # Preflight: refuse to overwrite any existing file whose content differs.
    for path, expected in files:
        if path.exists() and path.read_text(encoding="utf-8-sig") != expected:
            raise FileExistsError(f"Existing file differs; nothing overwritten: {path}")
    for path, expected in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as f:
                f.write(expected)
            print("CREATED:", path)
        else:
            print("UNCHANGED:", path)
    print("\nPilot: 3 blocks x 11 conditions = 33 independent Jitsi sessions")
    print("Video SHA256:", manifest["video_sha256"])
    print("Available disk space: %.1f GiB" % (shutil.disk_usage(root).free / 1024**3))
    print("No Jitsi sessions started. Existing jobs and videos unchanged.")
    print("\nSTART FIRST BLOCK (only after reviewing the configs):")
    print("python 12_visual_oracle_runner.py --job captures\\dashboard_jobs\\job_exp04_pilot_shields_b01")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    plan(args.root.resolve(), args.seed)


if __name__ == "__main__":
    main()
