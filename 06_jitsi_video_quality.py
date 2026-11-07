"""Offline exploratory reference-vs-received video quality analysis.

Run from Windows Anaconda Prompt in D:\\visual-oracle-revision:
  python 06_jitsi_video_quality.py

Requires: numpy, opencv-python. Uses the Y4M source already in the project
folder and the latest captures/shields_* run. No Docker or Jitsi needed.

IMPORTANT: Searches for the best reference frame independently for each PNG.
The resulting SSIM/PSNR values are OPTIMISTIC exploratory diagnostics, not
unbiased test-oracle accuracy or proof of correct temporal synchronization.
"""

import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path('720p50_shields_ter.y4m')
CAPTURES_ROOT = Path('captures')
THUMB_W, THUMB_H = 160, 90
TOP_CANDIDATES = 12
MAX_FRAMES = 5000


def get_latest_run():
    runs = sorted(p for p in CAPTURES_ROOT.glob('shields_*') if p.is_dir())
    if not runs:
        raise FileNotFoundError('captures\\shields_* folder not found')
    # Prefer the most recent run containing actual remote frames.
    for run in reversed(runs):
        if list(run.glob('Participant_B_remote_*.png')):
            return run
    raise FileNotFoundError('Participant_B_remote_*.png not found')


def parse_y4m_header(header):
    parts = header.decode('ascii', errors='strict').strip().split()
    if not parts or parts[0] != 'YUV4MPEG2':
        raise ValueError('Y4M basligi gecersiz')
    props = {p[0]: p[1:] for p in parts[1:] if len(p) > 1}
    width, height = int(props['W']), int(props['H'])
    fps_num, fps_den = map(int, props.get('F', '25:1').split(':'))
    chroma = props.get('C', '420')
    if not chroma.startswith('420') or 'p10' in chroma or 'p12' in chroma or 'p16' in chroma:
        raise ValueError(f'Yalnizca 8-bit YUV420 Y4M destekleniyor: {chroma}')
    if width % 2 or height % 2 or fps_num <= 0 or fps_den <= 0:
        raise ValueError('Gecersiz Y4M boyut/FPS')
    return width, height, fps_num / fps_den, chroma


def scan_y4m(path):
    """Read luma thumbnails and record offsets for random-access full frames."""
    offsets, thumbs = [], []
    with path.open('rb') as f:
        width, height, fps, chroma = parse_y4m_header(f.readline())
        frame_bytes = width * height * 3 // 2
        while len(offsets) < MAX_FRAMES:
            marker = f.readline()
            if not marker:
                break
            if not marker.startswith(b'FRAME') or marker[5:6] not in (b'\n', b' ', b'\r'):
                raise ValueError(f'Beklenmeyen Y4M frame basligi: {marker[:60]!r}')
            frame_offset = f.tell()
            raw_y = f.read(width * height)
            if len(raw_y) != width * height:
                raise ValueError('Eksik Y4M Y duzlemi')
            luma = np.frombuffer(raw_y, dtype=np.uint8).reshape(height, width)
            thumbs.append(cv2.resize(luma, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).copy())
            offsets.append(frame_offset)
            f.seek(frame_offset + frame_bytes)
        if not offsets:
            raise ValueError('Y4M dosyasinda hic kare yok')
        if len(offsets) == MAX_FRAMES:
            print(f'UYARI: Ilk {MAX_FRAMES} kaynak kare inceleniyor.')
    return width, height, fps, chroma, offsets, np.stack(thumbs)


def read_reference_frame(handle, offset, width, height, target_size):
    handle.seek(offset)
    n = width * height * 3 // 2
    payload = handle.read(n)
    if len(payload) != n:
        raise ValueError('Kaynak Y4M karesi okunamadi')
    yuv = np.frombuffer(payload, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    if (width, height) != target_size:
        bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    return bgr


def grayscale_ssim(a, b):
    """Standard Gaussian-window luminance SSIM, mean over pixels."""
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    var_x = cv2.GaussianBlur(x*x, (11, 11), 1.5) - mu_x*mu_x
    var_y = cv2.GaussianBlur(y*y, (11, 11), 1.5) - mu_y*mu_y
    cov = cv2.GaussianBlur(x*y, (11, 11), 1.5) - mu_x*mu_y
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    numerator = (2*mu_x*mu_y + c1) * (2*cov + c2)
    denominator = (mu_x*mu_x + mu_y*mu_y + c1) * (var_x + var_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-10)))


def psnr_bgr(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff*diff))
    return float('inf') if mse == 0 else 10 * math.log10((255.0**2)/mse)


def load_log(run):
    log = run / 'capture_log.csv'
    if not log.exists():
        return {}
    with log.open('r', encoding='utf-8-sig', newline='') as f:
        return {row['filename']: row for row in csv.DictReader(f) if row.get('filename')}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Source not found: {SOURCE.resolve()}')
    run = get_latest_run()
    png_files = sorted(run.glob('Participant_B_remote_*.png'))
    print('Kaynak:', SOURCE.resolve())
    print('Alinan kareler:', run.resolve())
    width, height, fps, chroma, offsets, thumbs = scan_y4m(SOURCE)
    print(f'Y4M: {width}x{height}, {fps:.3f} FPS, {len(offsets)} kare, '
          f'{len(offsets)/fps:.3f} saniye, chroma={chroma}')
    if len(offsets) / fps < 10:
        print('UYARI: Kaynak 10 saniyeden kisa; Chrome tekrar oynatiyor olabilir.')
    log = load_log(run)
    rows = []
    out_csv = run / 'quality_exploratory.csv'
    with SOURCE.open('rb') as f:
        for png in png_files:
            received = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if received is None:
                print('HATA: PNG okunamadi:', png)
                continue
            h, w = received.shape[:2]
            received_gray = cv2.cvtColor(received, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(received_gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            # Fast shortlist based on luminance MSE at low resolution.
            diff = thumbs.astype(np.float32) - thumb.astype(np.float32)
            coarse = np.mean(diff * diff, axis=(1, 2))
            candidates = np.argsort(coarse)[:min(TOP_CANDIDATES, len(offsets))]
            best = None
            best_frame = None
            for index in candidates:
                reference = read_reference_frame(f, offsets[int(index)], width, height, (w, h))
                score = grayscale_ssim(reference, received)
                if best is None or score > best['ssim_gray']:
                    best = {
                        'source_frame_index': int(index),
                        'source_time_s': round(float(index) / fps, 5),
                        'ssim_gray': round(score, 6),
                        'psnr_bgr_db': round(psnr_bgr(reference, received), 4),
                        'coarse_mse': round(float(coarse[index]), 3),
                    }
                    best_frame = reference
            record = log.get(png.name, {})
            row = {
                'received_png': png.name,
                'received_width': w,
                'received_height': h,
                'video_current_time_s': record.get('videoCurrentTime', ''),
                'total_video_frames': record.get('totalVideoFrames', ''),
                'dropped_video_frames': record.get('droppedVideoFrames', ''),
                **best,
            }
            rows.append(row)
            # Visual comparison for the first successfully matched frame.
            if len(rows) == 1:
                comparison = np.hstack((best_frame, received))
                comparison_file = run / 'reference_vs_received.png'
                cv2.imwrite(str(comparison_file), comparison)
                print('Comparison image:', comparison_file)
            print(f"{png.name}: ref={best['source_frame_index']} "
                  f"(t={best['source_time_s']:.3f}s) "
                  f"SSIM={best['ssim_gray']:.4f} "
                  f"PSNR={best['psnr_bgr_db']:.2f} dB")
    if not rows:
        raise RuntimeError('No comparable PNG found')
    with out_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_ssim = float(np.mean([r['ssim_gray'] for r in rows]))
    print(f'\nOrtalama en-iyi-eslesme SSIM: {mean_ssim:.4f}')
    print('CSV:', out_csv.resolve())
    print('UYARI: Bu en-iyi-eslesme aramasi SSIM sonucunu yukari yanli yapar.')
    print('Bulgular kesif amaclidir; sabit zaman hizalamali deney degildir.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('HATA:', exc, file=sys.stderr)
        sys.exit(1)
