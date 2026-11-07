"""08 - Frame-ID-synchronized quality analysis for numbered Jitsi Y4M captures.

Windows / Anaconda Prompt (run in D:\\visual-oracle-revision):
    python 08_jitsi_synchronized_quality.py
    python 08_jitsi_synchronized_quality.py --run captures\\shields_YYYYMMDD_HHMMSS

Requirements: numpy, opencv-python. No Docker or Jitsi required for analysis.

The 24-cell barcode is decoded INDEPENDENTLY of the image-quality metric:
  1010 guard + 16-bit frame index (MSB-first) + 4-bit nibble-XOR checksum.
The same numbered source MUST have been used for the capture. The unnumbered
Y4M is used as reference because the original content outside the overlay is
identical. All pixels near the overlay are excluded from SSIM and PSNR.

Caveats: this reports quality for sparse captured frames, not per-frame
stream quality, latency, network throughput, or a population estimate.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ORIGINAL = Path('720p50_shields_ter.y4m')
METADATA = Path('720p50_shields_ter_numbered_metadata.json')
CAPTURES = Path('captures')
EXPECTED_NUMBERED = '720p50_shields_ter_numbered.y4m'
OVERLAY_MARGIN_PX = 8  # in received-image pixels, also protects SSIM's 11x11 window


def parse_header(line):
    parts = line.decode('ascii', 'strict').strip().split()
    if not parts or parts[0] != 'YUV4MPEG2':
        raise ValueError('Y4M basligi gecersiz')
    props = {p[0]: p[1:] for p in parts[1:] if len(p) > 1}
    w, h = int(props['W']), int(props['H'])
    n, d = map(int, props['F'].split(':'))
    chroma = props.get('C', '420')
    if not chroma.startswith('420') or any(x in chroma for x in ('p10', 'p12', 'p16')):
        raise ValueError(f'Yalnizca 8-bit planar YUV420 destekleniyor: {chroma}')
    if w % 2 or h % 2 or n <= 0 or d <= 0:
        raise ValueError('Gecersiz Y4M cozunurluk/FPS')
    return w, h, n / d


def index_y4m(path):
    offsets = []
    with path.open('rb') as f:
        w, h, fps = parse_header(f.readline())
        frame_bytes = w * h * 3 // 2
        while True:
            marker = f.readline()
            if not marker:
                break
            if not marker.startswith(b'FRAME') or marker[5:6] not in (b'\n', b'\r', b' '):
                raise ValueError(f'Gecersiz FRAME basligi: {marker[:50]!r}')
            offset = f.tell()
            offsets.append(offset)
            f.seek(offset + frame_bytes)
        if not offsets:
            raise ValueError('Kaynak Y4M kare icermiyor')
    return w, h, fps, offsets


def read_reference(f, offset, width, height, target_w, target_h):
    f.seek(offset)
    raw = f.read(width * height * 3 // 2)
    if len(raw) != width * height * 3 // 2:
        raise ValueError('Referans kare eksik okundu')
    yuv = np.frombuffer(raw, np.uint8).reshape(height * 3 // 2, width)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    if (width, height) != (target_w, target_h):
        interpolation = cv2.INTER_AREA if target_w <= width and target_h <= height else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=interpolation)
    return bgr


def decode_barcode(bgr, metadata):
    """Return (frame_index, confidence, code) or raise ValueError.

    Samples 24 cells at source coordinates scaled to received resolution.
    Robust median sampling over each cell's center avoids compressed edges.
    """
    h, w = bgr.shape[:2]
    src_w, src_h = int(metadata['width']), int(metadata['height'])
    cfg = metadata['barcode']
    x0 = float(cfg['source_cell_x'])
    y0 = float(cfg['source_cell_y'])
    cw = float(cfg['source_cell_width'])
    ch = float(cfg['source_cell_height'])
    count = int(cfg['number_of_cells'])
    if count != 24:
        raise ValueError(f'Beklenmeyen barkod hucre sayisi: {count}')
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    values = []
    for i in range(count):
        # Central 35% of each cell, away from resampling/codec edge artifacts.
        xa = int(round((x0 + (i + 0.325) * cw) * w / src_w))
        xb = int(round((x0 + (i + 0.675) * cw) * w / src_w))
        ya = int(round((y0 + 0.325 * ch) * h / src_h))
        yb = int(round((y0 + 0.675 * ch) * h / src_h))
        xa, xb = max(0, xa), min(w, max(xa + 1, xb))
        ya, yb = max(0, ya), min(h, max(ya + 1, yb))
        if xa >= w or ya >= h:
            raise ValueError('Barcode coordinates are outside image bounds')
        values.append(float(np.median(gray[ya:yb, xa:xb])))
    low, high = min(values), max(values)
    if high - low < 70:
        raise ValueError(f'Barkod kontrasti dusuk: {high-low:.1f}')
    threshold = (high + low) / 2
    code = ''.join('1' if v > threshold else '0' for v in values)
    if code[:4] != '1010':
        raise ValueError(f'Barkod koruma bitleri gecersiz: {code[:4]}')
    idx = int(code[4:20], 2)
    checksum = int(code[20:24], 2)
    expected = ((idx >> 12) ^ (idx >> 8) ^ (idx >> 4) ^ idx) & 15
    if checksum != expected:
        raise ValueError(f'Barkod checksum hatasi: {checksum} != {expected}')
    margin = min(abs(v - threshold) for v in values)
    return idx, round(margin, 2), code


def valid_pixel_mask(shape, metadata):
    h, w = shape[:2]
    src_w, src_h = int(metadata['width']), int(metadata['height'])
    x0, y0, x1, y1 = metadata['overlay_rect_xyxy_source_pixels']
    xa = max(0, math.floor(x0 * w / src_w) - OVERLAY_MARGIN_PX)
    ya = max(0, math.floor(y0 * h / src_h) - OVERLAY_MARGIN_PX)
    xb = min(w, math.ceil(x1 * w / src_w) + OVERLAY_MARGIN_PX)
    yb = min(h, math.ceil(y1 * h / src_h) + OVERLAY_MARGIN_PX)
    mask = np.ones((h, w), dtype=bool)
    mask[ya:yb, xa:xb] = False
    if mask.sum() < h * w * .5:
        raise ValueError('Maske goruntunun yarisindan fazlasini kapliyor')
    return mask, (xa, ya, xb, yb)


def quality_metrics(ref, received, mask):
    """Grayscale Gaussian SSIM and 3-channel RGB PSNR, overlay excluded."""
    x = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.cvtColor(received, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sig_x = cv2.GaussianBlur(x*x, (11, 11), 1.5) - mu_x*mu_x
    sig_y = cv2.GaussianBlur(y*y, (11, 11), 1.5) - mu_y*mu_y
    sig_xy = cv2.GaussianBlur(x*y, (11, 11), 1.5) - mu_x*mu_y
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    ssim_map = ((2*mu_x*mu_y+c1)*(2*sig_xy+c2)) / np.maximum(
        (mu_x**2+mu_y**2+c1)*(sig_x+sig_y+c2), 1e-9)
    ssim = float(np.mean(ssim_map[mask]))
    diff = ref.astype(np.float64) - received.astype(np.float64)
    mse = float(np.mean((diff[mask])**2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2/mse)
    return ssim, psnr


def find_run(explicit):
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f'Directory not found: {p}')
        runs = [p]
    else:
        runs = sorted((p for p in CAPTURES.glob('shields_*') if p.is_dir()), reverse=True)
    for p in runs:
        settings = p / 'settings.json'
        if not settings.exists():
            continue
        config = json.loads(settings.read_text(encoding='utf-8-sig'))
        video_name = str(config.get('source_video', '')).replace('\\', '/').split('/')[-1]
        if video_name != EXPECTED_NUMBERED:
            if explicit:
                raise ValueError(f'Bu kayit numarali kaynaktan alinmamis: {video_name}')
            continue
        if any(p.glob('Participant_B_remote_*.png')):
            return p
    raise FileNotFoundError(
        'No capture directory using a numbered Y4M was found. '
        '05_jitsi_video_capture.py icindeki Y4M_PATH_IN_CONTAINER ayarini '
        'numbered.y4m yapip yeniden calistirin.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='Analiz edilecek captures\\shields_... klasoru')
    args = parser.parse_args()
    for p in (ORIGINAL, METADATA):
        if not p.is_file():
            raise FileNotFoundError(f'Required file not found: {p.resolve()}')
    metadata = json.loads(METADATA.read_text(encoding='utf-8-sig'))
    run = find_run(args.run)
    w, h, fps, offsets = index_y4m(ORIGINAL)
    if (w, h) != (int(metadata['width']), int(metadata['height'])):
        raise ValueError('Metadata ile kaynak Y4M boyutlari farkli')
    if abs(fps - float(metadata['fps'])) > 1e-6:
        raise ValueError('Metadata ile kaynak Y4M FPS farkli')
    if len(offsets) != int(metadata['frame_count']):
        raise ValueError('Metadata and source Y4M frame counts differ')
    files = sorted(run.glob('Participant_B_remote_*.png'))
    print('Referans:', ORIGINAL.resolve())
    print('Numarali kaynak:', EXPECTED_NUMBERED)
    print('Kayit klasoru:', run.resolve())
    print(f'Referans: {w}x{h}, {fps:.3f} FPS, {len(offsets)} kare')
    rows, example = [], None
    with ORIGINAL.open('rb') as handle:
        for png in files:
            row = {'filename': png.name, 'status': 'ERROR', 'frame_index': '',
                   'reference_time_seconds': '', 'received_width': '',
                   'received_height': '', 'barcode_confidence': '',
                   'ssim_gray_masked': '', 'psnr_rgb_masked_db': '',
                   'valid_pixel_percent': '', 'error': ''}
            try:
                received = cv2.imread(str(png), cv2.IMREAD_COLOR)
                if received is None:
                    raise ValueError('PNG okunamadi')
                rh, rw = received.shape[:2]
                row['received_width'], row['received_height'] = rw, rh
                idx, confidence, code = decode_barcode(received, metadata)
                if idx >= len(offsets):
                    raise ValueError(f'Barkod kare indeksi kaynak disinda: {idx}')
                reference = read_reference(handle, offsets[idx], w, h, rw, rh)
                mask, box = valid_pixel_mask(received.shape, metadata)
                ssim, psnr = quality_metrics(reference, received, mask)
                row.update(status='OK', frame_index=idx,
                           reference_time_seconds=round(idx/fps, 6),
                           barcode_confidence=confidence,
                           ssim_gray_masked=round(ssim, 6),
                           psnr_rgb_masked_db=round(psnr, 4),
                           valid_pixel_percent=round(100*float(np.mean(mask)), 2))
                print(f'{png.name}: FRAME {idx:06d} (t={idx/fps:.3f}s), '
                      f'{rw}x{rh}, SSIM={ssim:.4f}, PSNR={psnr:.2f} dB')
                if example is None:
                    # Comparison figure uses identical black rectangles on both sides.
                    ref_vis, rec_vis = reference.copy(), received.copy()
                    xa, ya, xb, yb = box
                    ref_vis[ya:yb, xa:xb] = 0
                    rec_vis[ya:yb, xa:xb] = 0
                    title_h = 46
                    composite = np.full((rh + title_h, rw * 2, 3), 245, np.uint8)
                    composite[title_h:, :rw] = ref_vis
                    composite[title_h:, rw:] = rec_vis
                    cv2.putText(composite, f'REFERENCE FRAME {idx:06d}', (12, 31),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
                    cv2.putText(composite, f'RECEIVED / SSIM {ssim:.4f}', (rw+12, 31),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
                    example = composite
            except Exception as exc:
                row['error'] = str(exc)
                print(f'{png.name}: HATA: {exc}')
            rows.append(row)
    output_csv = run / 'quality_synchronized.csv'
    with output_csv.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    if example is not None:
        path = run / 'reference_vs_received_synchronized.png'
        if not cv2.imwrite(str(path), example):
            raise OSError('Comparison PNG could not be saved')
        print('Comparison:', path.resolve())
    good = [r for r in rows if r['status'] == 'OK']
    summary = {
        'reference_y4m': str(ORIGINAL.resolve()),
        'numbered_video': EXPECTED_NUMBERED,
        'capture_dir': str(run.resolve()),
        'capture_count': len(rows),
        'valid_barcode_count': len(good),
        'failed_count': len(rows)-len(good),
        'mean_ssim_gray_masked': (float(np.mean([r['ssim_gray_masked'] for r in good])) if good else None),
        'mean_psnr_rgb_masked_db': (float(np.mean([r['psnr_rgb_masked_db'] for r in good])) if good else None),
        'method': '24-bit independent frame barcode, masked grayscale SSIM (11x11 Gaussian), masked RGB PSNR',
        'caveat': 'Sparse sampled frames, not latency/FPS measurement; do not compare numerically with prior unmasked best-match SSIM.',
    }
    summary_path = run / 'quality_synchronized_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSonuc: {len(good)}/{len(rows)} kare dogrulandi')
    if good:
        print(f'Ortalama MASKELI senkron SSIM: {summary["mean_ssim_gray_masked"]:.4f}')
        print(f'Ortalama MASKELI senkron PSNR: {summary["mean_psnr_rgb_masked_db"]:.2f} dB')
    print('CSV:', output_csv.resolve())
    print('Ozet:', summary_path.resolve())
    if not good:
        raise RuntimeError('No frames were barcode-verified; check files and settings')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'HATA: {exc}', file=sys.stderr)
        sys.exit(1)
