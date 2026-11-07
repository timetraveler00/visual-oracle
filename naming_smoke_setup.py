import json
from pathlib import Path

root = Path(r"D:\visual-oracle-revision")
source = root / "captures/dashboard_jobs/job_exp02_pilot_01/scenario.json"
target = root / "captures/dashboard_jobs/job_naming_smoke_01/scenario.json"

config = json.loads(source.read_text(encoding="utf-8-sig"))

config["name"] = "EXP03 – Capture Naming Smoke Test"
config["duration"] = 10
config["interval"] = 2
config["repeats"] = 1
config["conditions"] = [
    {"scenario": "baseline", "severity": "medium"}
]
config["perceptual_metrics"] = []

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Created:", target)