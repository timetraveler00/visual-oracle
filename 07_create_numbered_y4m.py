"""Create a frame-numbered Y4M source for independent temporal alignment.

Windows / Anaconda Prompt:
    cd /d D:\\visual-oracle-revision
    python 07_create_numbered_y4m.py

Dependencies: numpy, opencv-python (already used by the quality scripts).
Output is an uncompressed YUV420 Y4M video with a machine-readable 24-cell
barcode and a human-readable frame index/timestamp. Video content outside the
fixed overlay rectangle is byte-for-byte unchanged in Y, U, and V planes.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SOURCE = '720p50_shields_ter.y4m'
DEFAULT_OUTPUT = '720p50_shields_ter_numbered.y4m'


def parse_header(line):
    parts = line.decode('ascii', 'strict').strip().split()
    if not parts or parts[0] != 'YUV4MPEG2':
        raise ValueError('YUV4MPEG2 header not found')
    tags = {p[0]: p[1:] for p in parts[1:] if len(p) > 1}
    width, height = int(tags['W']), int(tags['H'])
    fps_n, fps_d = map(int, tags['F'].split(':'))
    chroma = tags.get('C', '420')
    if width % 2 or height % 2 or fps_n <= 0 or fps_d <= 0:
        raise ValueError('Gecersiz cozunurluk veya FPS')
    if not chroma.startswith('420') or any(s in chroma for s in ('p10', 'p12', 'p16')):
        raise ValueError(f'Yalnizca 8-bit planar YUV420 destekleniyor: {chroma}')
    if width < 800 or height < 300:
        raise ValueError('Bu tasarim 720p icindir; kaynak boyutu cok kucuk')
    return width, height, fps_n, fps_d, chroma


def draw_overlay(y_plane, u_plane, v_plane, frame_idx, fps, width, height):
    # Fixed 560x116 region in source pixel coordinates, top left.
    # This region MUST be masked during reference-vs-received SSIM/PSNR.
    x0, y0, x1, y1 = 12, 12, 572, 128
    cv2.rectangle(y_plane, (x0, y0), (x1 - 1, y1 - 1), 16, -1)
    # Neutral chroma in the same area: black/white text and barcode.
    u_plane[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = 128
    v_plane[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = 128

    seconds = frame_idx / fps
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    label = f'FRAME {frame_idx:06d}  T {mins:02d}:{secs:06.3f}'
    cv2.putText(y_plane, label, (24, 58), cv2.FONT_HERSHEY_SIMPLEX,
                1.08, 235, 3, cv2.LINE_AA)

    # 4 guard bits + 16 frame-index bits + 4-bit XOR checksum.
    # Cell width = 20 px (10 px at 360p); cells are black or white.
    if frame_idx > 65535:
        raise ValueError('16-bit frame code exceeded (max 65535 frames)')
    checksum = ((frame_idx >> 12) ^ (frame_idx >> 8) ^
                (frame_idx >> 4) ^ frame_idx) & 0xF
    code = '1010' + format(frame_idx, '016b') + format(checksum, '04b')
    bx, by, cell_w, cell_h = 24, 78, 20, 32
    for i, bit in enumerate(code):
        left = bx + i * cell_w
        cv2.rectangle(y_plane, (left, by), (left + cell_w - 1, by + cell_h - 1),
                      235 if bit == '1' else 16, -1)
    return (x0, y0, x1, y1), label


def generate(source, output):
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f'Source not found: {source.resolve()}')
    if source.resolve() == output.resolve():
        raise ValueError('Source and output cannot be the same file')
    if output.exists():
        raise FileExistsError(f'Cikti zaten mevcut, ustune yazilmiyor: {output}')

    # A Y4M output is approximately the same size as its source.
    free = shutil.disk_usage(output.parent).free
    if free < source.stat().st_size * 1.15:
        raise OSError('Yeterli bos disk alani yok (en az kaynak boyutu kadar gerekli)')

    count = 0
    preview = output.with_name(output.stem + '_preview.png')
    metadata = output.with_name(output.stem + '_metadata.json')
    try:
        with source.open('rb') as src, output.open('xb') as dst:
            header = src.readline()
            width, height, fps_n, fps_d, chroma = parse_header(header)
            fps = fps_n / fps_d
            y_size = width * height
            uv_size = y_size // 4
            frame_size = y_size + 2 * uv_size
            dst.write(header)
            print(f'Kaynak: {width}x{height}, {fps:.3f} FPS, chroma={chroma}', flush=True)
            print(f'Cikti: {output.resolve()}', flush=True)
            while True:
                frame_marker = src.readline()
                if not frame_marker:
                    break
                if not frame_marker.startswith(b'FRAME') or frame_marker[5:6] not in (b'\n', b' ', b'\r'):
                    raise ValueError(f'Gecersiz FRAME isareti (kare {count}): {frame_marker[:60]!r}')
                payload = src.read(frame_size)
                if len(payload) != frame_size:
                    raise ValueError(f'Eksik kare verisi: {count}')
                # Copy into a writable buffer. Planes are I420: Y, U, V.
                buf = bytearray(payload)
                y = np.frombuffer(buf, dtype=np.uint8, count=y_size, offset=0).reshape(height, width)
                u = np.frombuffer(buf, dtype=np.uint8, count=uv_size, offset=y_size).reshape(height // 2, width // 2)
                v = np.frombuffer(buf, dtype=np.uint8, count=uv_size, offset=y_size + uv_size).reshape(height // 2, width // 2)
                overlay, label = draw_overlay(y, u, v, count, fps, width, height)
                if count == 0:
                    # OpenCV preview of the first overlaid YUV420 frame.
                    yuv = np.frombuffer(buf, dtype=np.uint8).reshape(height * 3 // 2, width)
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    if not cv2.imwrite(str(preview), bgr):
                        raise OSError('Onizleme PNG yazilamadi')
                dst.write(frame_marker)
                dst.write(buf)
                count += 1
                if count % 100 == 0:
                    print(f'{count} kare islendi...', flush=True)
        if not count:
            raise ValueError('No frames found in source')
        info = {
            'source': str(source.resolve()),
            'output': str(output.resolve()),
            'width': width, 'height': height,
            'fps_numerator': fps_n, 'fps_denominator': fps_d,
            'fps': fps, 'frame_count': count,
            'duration_seconds': count / fps,
            'chroma': chroma,
            'frame_number_base': 0,
            'overlay_rect_xyxy_source_pixels': list(overlay),
            'overlay_rect_normalized_xyxy': [overlay[0]/width, overlay[1]/height,
                                             overlay[2]/width, overlay[3]/height],
            'barcode': {
                'bits': '1010 + 16-bit frame index (MSB first) + 4-bit XOR of nibbles',
                'source_cell_x': 24, 'source_cell_y': 78,
                'source_cell_width': 20, 'source_cell_height': 32,
                'number_of_cells': 24,
            },
            'warning': 'Exclude overlay rectangle from SSIM/PSNR and other image-quality metrics.',
        }
        metadata.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nTAMAMLANDI: {count} kare, {count/fps:.3f} saniye', flush=True)
        print('Y4M:', output.resolve())
        print('Onizleme:', preview.resolve())
        print('Metadata:', metadata.resolve())
        print('NOT: Chrome kamerasi kaynak dosyayi basa sarabilir; frame numarasi her dongude tekrarlanir.')
    except Exception:
        if output.exists():
            output.unlink()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Y4M dosyasina kare numarasi, zaman ve barkod ekle')
    parser.add_argument('--source', default=DEFAULT_SOURCE)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        generate(args.source, args.output)
    except Exception as e:
        print(f'HATA: {e}', file=sys.stderr)
        sys.exit(1)
