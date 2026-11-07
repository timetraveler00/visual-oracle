#!/usr/bin/env python3
"""EXP04 Shields freeze/low: source-Y4M freeze validation (frames 150..224).

Requires imageio-ffmpeg (preferred, bundled FFmpeg) or working ffmpeg on PATH. Pillow is optional for a contact sheet.
Produces a frame-to-frame difference CSV, a text report, and selected PNGs.
"""
from __future__ import annotations
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

FRAME_LIST = [145, 149, 150, 151, 170, 190, 223, 224, 225, 230]
W, H = 320, 180  # only for the numerical analysis; source PNGs remain full size


def run(cmd, **kwargs):
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed (exit {e.returncode}): {' '.join(map(str, cmd))}") from e


def find_file(project: Path, kind: str):
    if kind == "fault":
        base = project / "fault_sources"
        matches = [p for p in base.rglob("*.y4m")
                   if "freeze" in p.name.lower() and "low" in p.name.lower()]
        shields = [p for p in matches if "shields" in p.name.lower()]
        if shields:
            matches = shields
    else:
        base = project / "videos"
        matches = [p for p in base.rglob("*.y4m")
                   if "shields" in p.name.lower() and "numbered" in p.name.lower()]
    matches = sorted(matches)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    print(f"Multiple {kind} files found. Choose the correct one:")
    for i, p in enumerate(matches, 1):
        print(f"  {i}. {p}")
    while True:
        try:
            selection = input(f"Number (1-{len(matches)}): ").strip()
        except EOFError:
            raise RuntimeError(f"Specify --{kind} with the exact file path")
        if selection.isdigit() and 1 <= int(selection) <= len(matches):
            return matches[int(selection) - 1]


def read_gray_frames(ffmpeg: str, video: Path, start: int, end: int):
    # The select filter uses zero-based video-frame indices, not timestamps.
    filt = f"select=between(n\\,{start}\\,{end}),scale={W}:{H},format=gray"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", filt, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = W * H
    frames = []
    try:
        for _ in range(end - start + 1):
            buf = bytearray()
            while len(buf) < frame_size:
                part = proc.stdout.read(frame_size - len(buf))
                if not part:
                    break
                buf.extend(part)
            if len(buf) != frame_size:
                break
            frames.append(bytes(buf))
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg decoding failed: {stderr}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if len(frames) != end - start + 1:
        raise RuntimeError(f"Expected {end-start+1} frames, got {len(frames)}. "
                           "Check the video length and the frame range.")
    return frames


def mae(a: bytes, b: bytes):
    return sum(abs(x-y) for x, y in zip(a, b)) / len(a)


def extract_pngs(ffmpeg: str, video: Path, dest: Path, label: str, selected: list[int]):
    dest.mkdir(parents=True, exist_ok=True)
    filt = "select=" + "+".join(f"eq(n\\,{n})" for n in selected)
    template = str(dest / (label + "_%03d.png"))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
           "-vf", filt, "-vsync", "0", "-start_number", "0", template]
    run(cmd)
    for i, n in enumerate(selected):
        src = dest / f"{label}_{i:03d}.png"
        if not src.exists():
            raise RuntimeError(f"Frame {n} was not extracted from {video}")
        src.rename(dest / f"{label}_{n:03d}.png")


def make_sheet(out: Path, selected: list[int], has_ref: bool):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed: skipping contact sheet (individual PNGs were saved).")
        return
    labels = ["reference", "fault"] if has_ref else ["fault"]
    thumb_w, thumb_h = 320, 180
    margin, caption = 12, 28
    cols = 5
    rows = ((len(selected) + cols - 1) // cols) * len(labels)
    canvas = Image.new("RGB", (cols * (thumb_w + margin) + margin,
                               rows * (thumb_h + caption + margin) + margin), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    for k, label in enumerate(labels):
        for j, n in enumerate(selected):
            f = out / f"{label}_{n:03d}.png"
            with Image.open(f) as im:
                thumb = im.convert("RGB").resize((thumb_w, thumb_h))
            x = margin + (j % cols) * (thumb_w + margin)
            y = margin + (k * ((len(selected) + cols - 1)//cols) + j // cols) * (thumb_h + caption + margin)
            canvas.paste(thumb, (x, y))
            draw.text((x+4, y+thumb_h+5), f"{label.upper()} - frame {n}", fill="black")
    canvas.save(out / "freeze_low_contact_sheet.png")


def select_ffmpeg(explicit: str | None = None) -> str:
    """Prefer imageio-ffmpeg's bundled standalone executable to conda's DLL-linked one."""
    if explicit:
        exe = explicit
    else:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "FFmpeg not found. Run: python -m pip install imageio-ffmpeg"
        )
    try:
        p = subprocess.run([exe, "-version"], capture_output=True, text=True,
                           timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Cannot start FFmpeg at {exe}: {exc}. "
            "Install the standalone binary with: python -m pip install imageio-ffmpeg"
        ) from exc
    if p.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed at {exe} (exit {p.returncode}). "
            "This may be the conda DLL mismatch. "
            "Run: python -m pip install imageio-ffmpeg"
        )
    print("FFMPEG:   ", exe)
    return exe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(r"D:\visual-oracle-revision"))
    parser.add_argument("--ffmpeg", type=str, help="Optional exact FFmpeg executable path")
    parser.add_argument("--fault", type=Path, help="Exact freeze/low Y4M file (optional)")
    parser.add_argument("--reference", type=Path, help="Clean numbered Shields Y4M file (optional)")
    parser.add_argument("--start", type=int, default=150)
    parser.add_argument("--end", type=int, default=224)
    parser.add_argument("--out", type=Path, help="Output folder (optional)")
    args = parser.parse_args()
    ffmpeg = select_ffmpeg(args.ffmpeg)
    if args.start < 1 or args.end <= args.start:
        sys.exit("ERROR: expected 1 <= start < end")
    fault = args.fault or find_file(args.project, "fault")
    reference = args.reference or find_file(args.project, "reference")
    if not fault or not fault.is_file():
        sys.exit("ERROR: freeze/low Y4M not found. Run with --fault FULL_PATH_TO_Y4M")
    if reference and not reference.is_file():
        sys.exit("ERROR: reference Y4M path does not exist")
    out = args.out or (args.project / "analysis" / "exp04_freeze_low_y4m_check")
    out.mkdir(parents=True, exist_ok=True)
    start, end = args.start, args.end
    lo, hi = max(0, start - 5), end + 6
    print("FAULT:    ", fault)
    print("REFERENCE:", reference if reference else "not found; fault-only analysis")
    print("OUTPUT:   ", out)
    print(f"Reading frames {lo}..{hi} at {W}x{H} for pixel-level analysis ...")
    f_frames = read_gray_frames(ffmpeg, fault, lo, hi)
    r_frames = read_gray_frames(ffmpeg, reference, lo, hi) if reference else None
    rows = []
    for n in range(lo, hi+1):
        idx = n - lo
        adjacent = mae(f_frames[idx], f_frames[idx-1]) if idx else None
        original = mae(f_frames[idx], r_frames[idx]) if r_frames else None
        rows.append({"frame": n,
                     "time_s_at_50fps": round(n / 50.0, 3),
                     "region": "freeze" if start <= n <= end else "outside",
                     "adjacent_mae_0_255": round(adjacent, 6) if adjacent is not None else "",
                     "fault_vs_reference_mae_0_255": round(original, 6) if original is not None else ""})
    csv_path = out / "freeze_low_frame_differences.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inside = [r["adjacent_mae_0_255"] for r in rows if start+1 <= r["frame"] <= end]
    outside = [r["adjacent_mae_0_255"] for r in rows if r["region"] == "outside" and r["adjacent_mae_0_255"] != ""]
    repeated = sum(v == 0 for v in inside)
    print(f"Within freeze interval: {repeated}/{len(inside)} consecutive frame pairs are PIXEL-IDENTICAL")
    print(f"Mean adjacent-frame MAE inside: {sum(inside)/len(inside):.6f} / 255")
    print(f"Mean adjacent-frame MAE outside: {sum(outside)/len(outside):.6f} / 255")
    if r_frames:
        outside_ref = [r["fault_vs_reference_mae_0_255"] for r in rows if r["region"] == "outside"]
        inside_ref = [r["fault_vs_reference_mae_0_255"] for r in rows if r["region"] == "freeze"]
        print(f"Fault-vs-clean MAE inside: {sum(inside_ref)/len(inside_ref):.6f} / 255")
        print(f"Fault-vs-clean MAE outside: {sum(outside_ref)/len(outside_ref):.6f} / 255")
    if repeated == len(inside):
        verdict = "Full-frame freeze CONFIRMED in the source Y4M for the entire selected interval."
    elif repeated > 0:
        verdict = "Some repeated frames found, but NOT a continuous full-frame freeze over the entire interval. Inspect PNGs and the injection specification."
    else:
        verdict = "No pixel-identical adjacent frames in the selected interval. A full-frame hold is NOT confirmed; check the actual fault mechanism and indices."
    print("VERDICT:", verdict)
    report = ["EXP04 Shields freeze/low source-Y4M check", "",
              f"Fault: {fault}", f"Reference: {reference or '(not found)'}",
              f"Range: {start}..{end} inclusive (zero-based), at nominal 50 FPS",
              f"Duration: {(end-start+1)/50:.3f} seconds",
              f"Consecutive pairs in region: {len(inside)}",
              f"Pixel-identical pairs: {repeated}/{len(inside)}",
              f"Mean adjacent MAE in region: {sum(inside)/len(inside):.6f} / 255",
              f"Mean adjacent MAE outside: {sum(outside)/len(outside):.6f} / 255",
              f"Verdict: {verdict}", "",
              "IMPORTANT: This tests the sender-side fault Y4M, NOT the received WebRTC video.",
              "A repeated frame number in a looped video is not sufficient proof of a freeze.",
              "Adjacent pixel differences near zero across the full intended interval are the key test."]
    if r_frames:
        report += [f"Mean fault-vs-clean MAE in region: {sum(inside_ref)/len(inside_ref):.6f} / 255",
                   f"Mean fault-vs-clean MAE outside: {sum(outside_ref)/len(outside_ref):.6f} / 255"]
    (out / "freeze_low_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    selected = sorted(set([n for n in FRAME_LIST if lo <= n <= hi] + [start, end]))
    print("Extracting full-resolution PNGs ...")
    extract_pngs(ffmpeg, fault, out, "fault", selected)
    if reference:
        extract_pngs(ffmpeg, reference, out, "reference", selected)
    make_sheet(out, selected, bool(reference))
    print("DONE. Please upload: freeze_low_report.txt, freeze_low_frame_differences.csv, freeze_low_contact_sheet.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
