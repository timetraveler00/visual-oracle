"""EXP04 Shields freeze/low: validate scene hold while ignoring the changing frame counter.

Read-only check. Requires numpy and imageio-ffmpeg.
Output: analysis/exp04_freeze_low_y4m_check/freeze_low_masked_report.txt
        analysis/exp04_freeze_low_y4m_check/freeze_low_masked_pairs.csv
"""
from pathlib import Path
import csv
import subprocess
import sys

import numpy as np
import imageio_ffmpeg

ROOT = Path(r"D:\visual-oracle-revision")
FAULT = ROOT / "fault_sources" / "720p50_shields_ter_2064530c7f" / "720p50_shields_ter_fault_freeze_low.y4m"
REFERENCE = ROOT / "videos" / "720p50_shields_ter_numbered.y4m"
OUT = ROOT / "analysis" / "exp04_freeze_low_y4m_check"
W, H = 320, 180
START, END = 150, 224  # zero-based, inclusive
# Top-left burned-in FRAME/T overlay; retain 89.8% of image pixels.
MASK_X_END, MASK_Y_END = 155, 38


def read_frames(path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path),
        "-vf", f"scale={W}:{H}:flags=neighbor",
        "-fps_mode", "passthrough",
        "-frames:v", str(END + 2),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed for {path}:\n{result.stderr.decode(errors='replace')}")
    frame_bytes = W * H * 3
    if len(result.stdout) % frame_bytes:
        raise RuntimeError(f"Unexpected decoded byte count: {len(result.stdout)}")
    arr = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, H, W, 3)
    if len(arr) <= END + 1:
        raise RuntimeError(f"Too few frames in {path}: {len(arr)}")
    return arr


def masked_metrics(a, b, keep):
    a_pixels = a[keep]
    b_pixels = b[keep]
    differences = np.abs(a_pixels.astype(np.int16) - b_pixels.astype(np.int16))
    changed_pixels = int(np.count_nonzero(np.any(a_pixels != b_pixels, axis=1)))
    return float(differences.mean()), changed_pixels


def main():
    for path in (FAULT, REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Y4M: {path}")
    print("FFMPEG:", imageio_ffmpeg.get_ffmpeg_exe())
    print("FAULT:", FAULT)
    print("REFERENCE:", REFERENCE)
    print(f"Decoding frames 0..{END+1} at {W}x{H} ...")
    fault = read_frames(FAULT)
    reference = read_frames(REFERENCE)

    keep = np.ones((H, W), dtype=bool)
    keep[:MASK_Y_END, :MASK_X_END] = False
    kept = int(keep.sum())
    print(f"Masked frame-counter area: x=0..{MASK_X_END-1}, y=0..{MASK_Y_END-1}")
    print(f"Analyzed pixels per frame: {kept}/{W*H} ({100*kept/(W*H):.2f}%)")

    rows = []
    for idx in range(START, END + 2):
        adj_mae, adj_changed = masked_metrics(fault[idx-1], fault[idx], keep)
        ref_mae, ref_changed = masked_metrics(fault[idx], reference[idx], keep)
        rows.append({
            "frame": idx,
            "region": "freeze" if idx <= END else "after_freeze",
            "masked_adjacent_mae_0_255": f"{adj_mae:.9f}",
            "masked_adjacent_changed_pixels": adj_changed,
            "masked_adjacent_identical": adj_changed == 0,
            "masked_fault_vs_reference_mae_0_255": f"{ref_mae:.9f}",
            "masked_fault_vs_reference_changed_pixels": ref_changed,
        })

    # Hold is tested on 151..224 (74 consecutive frame pairs).
    inside = rows[1:-1]
    identical = sum(int(r["masked_adjacent_identical"]) for r in inside)
    mean_inside = float(np.mean([float(r["masked_adjacent_mae_0_255"]) for r in inside]))
    start_ref_mae, start_ref_changed = masked_metrics(fault[START], reference[START], keep)
    end_ref_mae, end_ref_changed = masked_metrics(fault[END+1], reference[END+1], keep)
    jump_mae = float(rows[-1]["masked_adjacent_mae_0_255"])

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "freeze_low_masked_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "EXP04 Shields freeze/low – frame-counter-masked Y4M check",
        f"Fault: {FAULT}",
        f"Reference: {REFERENCE}",
        f"Frame range (zero-based): {START}..{END} inclusive",
        f"Masked rectangle at {W}x{H}: x=0..{MASK_X_END-1}, y=0..{MASK_Y_END-1}",
        f"Unmasked pixels: {kept}/{W*H} ({100*kept/(W*H):.2f}%)",
        f"Pixel-identical adjacent scene pairs inside freeze (151..224): {identical}/{END-START}",
        f"Mean masked adjacent MAE inside freeze: {mean_inside:.9f} / 255",
        f"At frame 150: fault vs reference masked MAE = {start_ref_mae:.9f}; changed pixels = {start_ref_changed}",
        f"At frame 225: fault vs reference masked MAE = {end_ref_mae:.9f}; changed pixels = {end_ref_changed}",
        f"Scene change at 224->225: masked adjacent MAE = {jump_mae:.9f} / 255",
        "VERDICT: " + (
            "SCENE HOLD CONFIRMED for all 74 adjacent pairs outside the frame-counter overlay."
            if identical == END - START else
            "Scene hold NOT fully pixel-identical after masking. Inspect the per-frame CSV and consider residual moving regions."
        ),
        "Caution: Sender-side Y4M only; this does not prove the receiver observed all 75 frames as a continuous freeze.",
    ]
    txt_path = OUT / "freeze_low_masked_report.txt"
    txt_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[5:]))
    print("\nSaved:", txt_path, "\n      ", csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
