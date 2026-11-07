#!/usr/bin/env python3
"""EXP04 video-source preparation for the existing visual-oracle Jitsi runner.

Inputs: Existing Shields, Xiph DERF FourPeople/Parkrun, and a deterministic synthetic screen-share.
Outputs: 1280x720, 50 FPS, 8-bit 4:2:0 Y4M in ROOT/videos/.

Does NOT execute Jitsi tests. Optional --make-smoke-jobs creates configs only.

Usage (Anaconda Prompt):
    cd /d D:\\visual-oracle-revision
    python exp04_prepare_videos.py --only shields --make-smoke-jobs
    python exp04_prepare_videos.py --only fourpeople
    python exp04_prepare_videos.py --only parkrun screenshare --make-smoke-jobs
    python exp04_prepare_videos.py --only all --make-smoke-jobs
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

SOURCES = {
    "shields": {
        "url": None,
        "name": "720p50_shields_ter.y4m",
        "role": "historical_reference_exp01_exp03",
        "origin": "Previously used Xiph DERF Shields source; reused unchanged for EXP04",
    },
    "fourpeople": {
        "url": "https://media.xiph.org/video/derf/y4m/FourPeople_1280x720_60.y4m",
        "name": "720p50_fourpeople_exp04.y4m",
        "role": "speakers_faces",
        "origin": "Xiph.org DERF test media, FourPeople, original 60 FPS",
    },
    "parkrun": {
        "url": "https://media.xiph.org/video/derf/y4m/720p50_parkrun_ter.y4m",
        "name": "720p50_parkrun_exp04.y4m",
        "role": "dynamic_motion",
        "origin": "Xiph.org DERF test media, Parkrun, original 50 FPS",
    },
    "screenshare": {
        "url": None,
        "name": "720p50_screenshare_exp04.y4m",
        "role": "synthetic_text_charts_ui",
        "origin": "Original deterministic synthetic screen-share-like video (not a real captured app)",
    },
}
W, H = 1280, 720
FPS = 50
DURATION = 10


def command_version(executable: str) -> str:
    result = subprocess.run([executable, "-version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=True)
    return result.stdout.splitlines()[0].strip()


def check_dependencies(selected: list[str]) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and FFprobe must both be available on PATH. Try: ffmpeg -version")
    if "screenshare" in selected:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Synthetic ScreenShare requires numpy and opencv-python: python -m pip install numpy opencv-python") from exc


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_y4m(path: Path, fps: int, duration: int, *, existing_source: bool = False) -> dict:
    """Check FFprobe metadata and the exact expected count of Y4M frames."""
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,pix_fmt",
        "-of", "json", str(path)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"FFprobe did not find one video stream: {path}")
    s = streams[0]
    if (s.get("width"), s.get("height")) != (W, H):
        raise RuntimeError(f"Unexpected geometry: {s}")
    if s.get("r_frame_rate") != f"{fps}/1":
        raise RuntimeError(f"Unexpected FPS: {s}")
    if s.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"Unexpected pixel format: {s}")
    # FFmpeg's yuv4mpegpipe writes one ASCII 'FRAME\\n' marker per frame.
    with path.open("rb") as f:
        header = f.readline()
        first_marker = f.readline()
    if not header.startswith(b"YUV4MPEG2 ") or first_marker != b"FRAME\n":
        raise RuntimeError(f"Unexpected Y4M header/first-frame marker: {path}")
    frame_bytes = W * H * 3 // 2
    expected_frames = fps * duration
    actual_size = path.stat().st_size
    per_frame_bytes = len(b"FRAME\n") + frame_bytes
    payload_bytes = actual_size - len(header)
    if payload_bytes < 0 or payload_bytes % per_frame_bytes:
        raise RuntimeError(f"Unexpected Y4M size/FRAME markers: {path.name}; "
                           "source may contain extended FRAME headers or be incomplete")
    actual_frames = payload_bytes // per_frame_bytes
    if existing_source:
        if actual_frames < expected_frames:
            raise RuntimeError(f"Shields has {actual_frames} frames, needs at least {expected_frames}")
    elif actual_frames != expected_frames:
        raise RuntimeError(f"Frame count mismatch: {actual_frames} != {expected_frames}: {path.name}")
    return {"width": W, "height": H, "fps": fps,
            "pixel_format": "yuv420p", "frame_count": actual_frames,
            "duration_seconds": round(actual_frames / fps, 5), "bytes": actual_size}


def run_ffmpeg_url(key: str, dest: Path, fps: int, duration: int) -> None:
    url = SOURCES[key]["url"]
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    # Process remote Y4M as a stream. Do not keep a second full-size source copy.
    filter_graph = f"fps={fps},scale={W}:{H}:flags=bicubic,format=yuv420p"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning", "-stats",
           "-y", "-i", url, "-vf", filter_graph, "-frames:v", str(fps * duration),
           "-an", "-f", "yuv4mpegpipe", str(part)]
    print(f"\n[{key}] Streaming source from Xiph; preparing {fps * duration} frames...", flush=True)
    try:
        subprocess.run(cmd, check=True)
        inspect_y4m(part, fps, duration)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def put(img, text, x, y, scale=0.65, color=(35, 45, 65), thickness=2):
    import cv2
    cv2.putText(img, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def screenshare_frame(i: int, fps: int, duration: int):
    """Create deterministic text/graph/UI content with ongoing pointer and scroll motion."""
    import cv2
    import numpy as np
    t = i / fps
    img = np.full((H, W, 3), (245, 247, 250), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (W, 82), (42, 56, 78), -1)
    put(img, "EXP04 / SCREEN SHARING", 35, 49, 0.85, (245, 245, 245), 2)
    put(img, f"t={t:05.2f}s", 1080, 49, 0.72, (210, 230, 255), 2)
    cv2.rectangle(img, (0, 82), (W, 119), (222, 229, 236), -1)
    put(img, "Synthetic UI test source  |  dynamic content  |  no injected faults", 35, 108, 0.54, (70, 75, 85), 1)
    # The existing numbered overlay covers x=12..572,y=12..128, which will be masked.
    if t < duration / 3:
        put(img, "Code editor - scrolling text and cursor", 45, 175, 0.88)
        cv2.rectangle(img, (44, 207), (1235, 647), (34, 43, 56), -1)
        lines = [
            "def visual_oracle(frame, reference):",
            "    aligned = synchronize(frame, reference)",
            "    ssim = compute_ssim(aligned, reference)",
            "    dists = compute_dists(aligned, reference)",
            "    return combine(ssim, dists)",
            "# Expected behavior: preserve content",
            "# Fault labels must not leak to oracle",
            "result = visual_oracle(frame, reference)",
            "print(result)",
            "# Run-level evaluation avoids leakage",
            "# Freeze detection needs motion",
        ]
        offset = int((t * 21) % 52)
        for n, line in enumerate(lines):
            y = 245 + n * 41 - offset
            if 235 <= y <= 620:
                put(img, f"{n+1:02d}", 58, y, 0.58, (120, 132, 148), 1)
                put(img, line, 120, y, 0.62, (208, 229, 241), 1)
        cx = 460 + int(110 * math.sin(1.8 * t))
        cy = 340 + int(90 * math.cos(1.3 * t))
    elif t < 2 * duration / 3:
        put(img, "Experiment dashboard - changing metrics", 45, 175, 0.88)
        cv2.rectangle(img, (45, 210), (1230, 648), (255, 255, 255), -1)
        cv2.rectangle(img, (45, 210), (1230, 648), (210, 216, 222), 2)
        put(img, "SSIM by experimental condition", 80, 258, 0.75)
        cv2.line(img, (135, 560), (1110, 560), (90, 98, 110), 2)
        cv2.line(img, (135, 300), (135, 560), (90, 98, 110), 2)
        labels = ["BASE", "BLUR-L", "BLUR-M", "BLUR-H"]
        for k, label in enumerate(labels):
            val = max(0.12, (0.87 - 0.13 * k) + 0.025 * math.sin(1.8 * t + k))
            bar_h = int(val * 230)
            x = 230 + 235 * k
            cv2.rectangle(img, (x, 560 - bar_h), (x + 120, 560), (198 - k * 25, 110 + k * 15, 48 + k * 22), -1)
            put(img, label, x + 8, 592, 0.55, (40, 45, 55), 1)
            put(img, f"{val:.3f}", x + 8, 547 - bar_h, 0.56, (40, 45, 55), 1)
        cx = 680 + int(170 * math.sin(t * 1.8))
        cy = 365 + int(85 * math.cos(t * 1.4))
    else:
        put(img, "Presentation - visual test requirements", 45, 175, 0.88)
        cv2.rectangle(img, (65, 210), (1210, 645), (255, 255, 255), -1)
        cv2.rectangle(img, (65, 210), (1210, 645), (215, 219, 226), 2)
        put(img, "Video conferencing test oracle", 125, 285, 1.02)
        bullets = [
            "1. Preserve semantic content",
            "2. Detect visual degradation",
            "3. Identify temporal interruptions",
            "4. Report false alarms separately",
        ]
        for k, bullet in enumerate(bullets):
            put(img, bullet, 140, 355 + k * 56, 0.75, (58, 67, 80), 2)
        # Animated timeline / pointer: intentional reference motion for freeze tests.
        progress = (t - 2 * duration / 3) / (duration / 3)
        cv2.rectangle(img, (140, 600), (1070, 615), (224, 228, 233), -1)
        cv2.rectangle(img, (140, 600), (140 + int(930 * progress), 615), (160, 110, 35), -1)
        cx = 720 + int(220 * math.sin(t * 1.7))
        cy = 360 + int(85 * math.cos(t * 1.4))
    # Always moving cursor, ensuring a small but observable temporal signal.
    pointer = np.array([[cx, cy], [cx + 3, cy + 25], [cx + 9, cy + 18],
                        [cx + 17, cy + 24], [cx + 21, cy + 19]], dtype=np.int32)
    cv2.fillPoly(img, [pointer], (20, 20, 20))
    cv2.polylines(img, [pointer], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (35, 679), (1245, 693), (220, 224, 230), -1)
    cv2.rectangle(img, (35, 679), (35 + int(1210 * (i + 1) / (fps * duration)), 693),
                  (70, 150, 105), -1)
    return img


def generate_screenshare(dest: Path, fps: int, duration: int) -> None:
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{W}x{H}",
           "-framerate", str(fps), "-i", "pipe:0", "-an", "-pix_fmt", "yuv420p",
           "-f", "yuv4mpegpipe", str(part)]
    print(f"\n[screenshare] Generating {fps * duration} synthetic frames...", flush=True)
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for i in range(fps * duration):
            process.stdin.write(screenshare_frame(i, fps, duration).tobytes())
            if (i + 1) % fps == 0:
                print(f"  synthetic second {(i + 1)//fps}/{duration}", flush=True)
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        returncode = process.wait()
        if returncode:
            raise RuntimeError("FFmpeg failed generating ScreenShare: " + stderr.decode("utf-8", "replace")[-1500:])
        inspect_y4m(part, fps, duration)
        part.replace(dest)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        part.unlink(missing_ok=True)
        if process.stderr is not None:
            process.stderr.close()


def make_smoke_job(root: Path, key: str, video_name: str, duration: int) -> Path:
    job_dir = root / "captures" / "dashboard_jobs" / f"job_exp04_smoke_{key}"
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "scenario.json"
    if config_path.exists():
        print(f"[SKIP] Existing smoke job config not overwritten: {config_path}")
        return config_path
    cfg = {
        "schema_version": 1,
        "name": f"EXP04 source validation - {key} (3 Jitsi sessions)",
        "video": f"videos/{video_name}",
        "node_count": 2, "duration": 30, "interval": 1, "repeats": 1,
        "clip_seconds": duration, "perceptual_metrics": [],
        "conditions": [
            {"scenario": "baseline", "severity": "medium"},
            {"scenario": "blur", "severity": "low"},
            {"scenario": "freeze", "severity": "medium"},
        ],
        "methodology": "Smoke test only. Independent Jitsi sessions; source-side faults. "
                       "Clip loops at source boundary; exclude loop transitions from temporal analyses. "
                       "Not used for confirmatory model comparison.",
    }
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Visual oracle project root (default: current directory)")
    p.add_argument("--only", nargs="+", choices=[*SOURCES, "all"], default=["all"], help="Videos to prepare")
    p.add_argument("--make-smoke-jobs", action="store_true", help="Write smoke-test scenario.json configs; DO NOT execute")
    p.add_argument("--force", action="store_true", help="Regenerate existing prepared videos")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    selected = list(SOURCES) if "all" in args.only else list(dict.fromkeys(args.only))
    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    check_dependencies(selected)
    ffmpeg_version = command_version("ffmpeg")
    manifest_path = root / "exp04_video_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise RuntimeError("Existing manifest is unreadable. Back it up; refusing to overwrite.")
    else:
        manifest = {"schema_version": 1, "project": "EXP04", "videos": {}}
    manifest["ffmpeg_version"] = ffmpeg_version
    manifest["normalization"] = {"width": W, "height": H, "fps": FPS,
                                  "pixel_format": "yuv420p", "duration_seconds": DURATION}
    # Existing Shields is reused without copying or modification.
    # Each newly generated 10 s Y4M is about 691 MB.
    free = shutil.disk_usage(videos_dir).free
    to_generate = [key for key in selected if key != "shields" and
                   (args.force or not (videos_dir / SOURCES[key]["name"]).exists())]
    required = len(to_generate) * (W * H * 3 // 2 * FPS * DURATION)
    if free < required + 1024 ** 3:
        raise OSError(f"Low free disk space: {free/1024**3:.1f} GiB; "
                      f"at least {(required+1024**3)/1024**3:.1f} GiB needed for preparation alone")
    for key in selected:
        dest = videos_dir / SOURCES[key]["name"]
        if key == "shields":
            if not dest.is_file():
                root_copy = root / dest.name
                hint = (f" Found a copy at {root_copy}; copy it to {dest} "
                        "without deleting the original." if root_copy.is_file() else "")
                raise FileNotFoundError(f"Existing Shields video is missing: {dest}.{hint}")
            if args.force:
                print("[shields] --force ignored: preserving original EXP01-EXP03 video")
            info = inspect_y4m(dest, FPS, DURATION, existing_source=True)
            print(f"\n[shields] Reusing historical source unchanged: {dest}", flush=True)
        elif dest.exists() and not args.force:
            print(f"\n[{key}] Existing output: validating, not overwriting: {dest}", flush=True)
            info = inspect_y4m(dest, FPS, DURATION)
        else:
            if dest.exists() and args.force:
                print(f"[{key}] --force: preparing replacement in a temporary file first")
                tmp_dest = dest.with_name(dest.stem + "_replacement" + dest.suffix)
                tmp_dest.unlink(missing_ok=True)
            else:
                tmp_dest = dest
            try:
                if key == "screenshare":
                    generate_screenshare(tmp_dest, FPS, DURATION)
                else:
                    run_ffmpeg_url(key, tmp_dest, FPS, DURATION)
                info = inspect_y4m(tmp_dest, FPS, DURATION)
                if tmp_dest != dest:
                    tmp_dest.replace(dest)
            finally:
                if tmp_dest != dest:
                    tmp_dest.unlink(missing_ok=True)
        digest = hash_file(dest)
        manifest["videos"][key] = {
            "filename": dest.name,
            "relative_path": f"videos/{dest.name}",
            "source": SOURCES[key]["url"],
            "origin": SOURCES[key]["origin"],
            "content_role": SOURCES[key]["role"],
            "sha256": digest,
            "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **info,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{key}] VERIFIED {info['frame_count']} frames | {info['bytes']/1024**2:.1f} MiB | SHA256 {digest[:16]}...", flush=True)
        if args.make_smoke_jobs:
            if key in ("shields", "fourpeople"):
                config = make_smoke_job(root, key, dest.name, DURATION)
                print(f"[{key}] Smoke config: {config}")
            else:
                print(f"[{key}] HOLDOUT: no smoke Jitsi job created before thresholds are frozen")
    print(f"\nDone. Manifest: {manifest_path}")
    print("No Jitsi sessions were started. No existing Shields source was modified.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
