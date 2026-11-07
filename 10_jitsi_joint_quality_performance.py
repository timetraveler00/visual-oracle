"""10 - Jitsi synchronized quality + receiver WebRTC performance in ONE run.

Windows Anaconda Prompt:
  cd /d D:\\visual-oracle-revision
  python 10_jitsi_joint_quality_performance.py

Requires 05_jitsi_video_capture.py, 08_jitsi_synchronized_quality.py,
09_jitsi_webrtc_stats.py in the same directory; selenium, opencv-python, numpy.
Source: numbered Shields Y4M installed in both Selenium Chrome nodes.
Outputs: captures/joint_YYYYMMDD_HHMMSS/{joint_measurements.csv,summary.json,
Participant_B_remote_*.png, raw_stats_*.json, Participant_B_ui.png, settings.json}.

Samples WebRTC getStats first, then captures the remote video immediately after.
Stats deltas describe the PRECEDING interval, not the instantaneous PNG frame.
Image quality is calculated OFFLINE after the 60-s capture run to avoid
loading the receiving Chrome / Selenium test while measuring performance.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def load_script(alias, filename):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f'Required file missing: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_script('jitsi_capture05', '05_jitsi_video_capture.py')
quality = load_script('jitsi_quality08', '08_jitsi_synchronized_quality.py')
rtc = load_script('jitsi_rtc09', '09_jitsi_webrtc_stats.py')

SOURCE = ROOT / '720p50_shields_ter.y4m'
METADATA = ROOT / '720p50_shields_ter_numbered_metadata.json'
NUMBERED = '720p50_shields_ter_numbered.y4m'
CONTAINER_VIDEO = '/tmp/' + NUMBERED
# Optional dashboard-provided identifier. Standalone execution keeps legacy naming.
RUN_NAME_PREFIX = ''

FIELDS = [
    'sample', 'elapsed_s', 'local_time', 'status', 'error',
    'capture_filename', 'capture_status', 'capture_error',
    'capture_width', 'capture_height', 'capture_bytes', 'capture_offset_ms',
    'frame_index', 'reference_time_s', 'barcode_confidence',
    'ssim_gray_masked', 'psnr_rgb_masked_db', 'valid_pixel_percent',
    'quality_status', 'quality_error',
    'inbound_id', 'codec', 'connection_state',
    'rtp_width', 'rtp_height', 'reported_fps', 'observed_decode_fps',
    'bitrate_mbps', 'interval_packet_loss_pct', 'jitter_ms',
    'interval_frames_dropped', 'interval_freezes',
    'playback_width', 'playback_height', 'playback_total_frames',
    'frames_decoded', 'packets_received', 'packets_lost', 'bytes_received',
    'stats_to_capture_ms',
]


def numeric(value):
    try:
        if value in ('', None):
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def avg(rows, key):
    vals = [numeric(r.get(key)) for r in rows]
    vals = [x for x in vals if x is not None]
    return round(statistics.mean(vals), 6) if vals else None


def safe_delta(new, old):
    return rtc.delta(new, old)


def compute_intervals(row, stream, previous, elapsed, previous_elapsed):
    if previous is None or previous_elapsed is None:
        return
    dt = elapsed - previous_elapsed
    if dt <= 0:
        return
    decoded = safe_delta(stream.get('framesDecoded'), previous.get('framesDecoded'))
    byte_delta = safe_delta(stream.get('bytesReceived'), previous.get('bytesReceived'))
    received = safe_delta(stream.get('packetsReceived'), previous.get('packetsReceived'))
    lost = safe_delta(stream.get('packetsLost'), previous.get('packetsLost'))
    row['observed_decode_fps'] = round(decoded / dt, 4) if decoded is not None else ''
    row['bitrate_mbps'] = round(byte_delta * 8 / dt / 1e6, 5) if byte_delta is not None else ''
    if received is not None and lost is not None and (received + lost) > 0:
        row['interval_packet_loss_pct'] = round(100 * lost / (received + lost), 5)
    drops = safe_delta(stream.get('framesDropped'), previous.get('framesDropped'))
    freezes = safe_delta(stream.get('freezeCount'), previous.get('freezeCount'))
    row['interval_frames_dropped'] = drops if drops is not None else ''
    row['interval_freezes'] = freezes if freezes is not None else ''


def enrich_quality(rows, run_dir):
    metadata = json.loads(METADATA.read_text(encoding='utf-8-sig'))
    w, h, fps, offsets = quality.index_y4m(SOURCE)
    if (w, h) != (int(metadata['width']), int(metadata['height'])):
        raise ValueError('Referans ve metadata boyutlari farkli')
    if abs(fps - float(metadata['fps'])) > 1e-6 or len(offsets) != int(metadata['frame_count']):
        raise ValueError('Reference FPS/frame count differs from metadata')
    with SOURCE.open('rb') as handle:
        for row in rows:
            if row['capture_status'] != 'OK':
                row['quality_status'] = 'SKIPPED'
                continue
            try:
                received = cv2.imread(str(run_dir / row['capture_filename']), cv2.IMREAD_COLOR)
                if received is None:
                    raise ValueError('PNG okunamadi')
                rh, rw = received.shape[:2]
                index, confidence, _ = quality.decode_barcode(received, metadata)
                if not (0 <= index < len(offsets)):
                    raise ValueError(f'Barkod kare indeksi kaynak disinda: {index}')
                reference = quality.read_reference(handle, offsets[index], w, h, rw, rh)
                mask, _ = quality.valid_pixel_mask(received.shape, metadata)
                ssim, psnr = quality.quality_metrics(reference, received, mask)
                row.update(frame_index=index, reference_time_s=round(index / fps, 5),
                           barcode_confidence=confidence,
                           ssim_gray_masked=round(ssim, 6),
                           psnr_rgb_masked_db=round(psnr, 4) if math.isfinite(psnr) else 'inf',
                           valid_pixel_percent=round(float(mask.mean()) * 100, 2),
                           quality_status='OK')
            except Exception as exc:
                row['quality_status'] = 'ERROR'
                row['quality_error'] = str(exc)
    return {'source_width': w, 'source_height': h, 'source_fps': fps,
            'source_frame_count': len(offsets)}


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=60, help='Olcum suresi, saniye (varsayilan 60)')
    parser.add_argument('--interval', type=int, default=5, help='Ornekleme araligi, saniye (varsayilan 5)')
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0 or args.duration % args.interval:
        parser.error('duration pozitif olmali ve interval degerine tam bolunmeli')
    for p in (SOURCE, METADATA):
        if not p.is_file():
            raise FileNotFoundError(f'Required file not found: {p}')
    # Reuse the already-tested 09 Chrome/RTCPeerConnection preload mechanism.
    rtc.Y4M_PATH_IN_CONTAINER = CONTAINER_VIDEO
    # Keep the joint_ prefix for compatibility with existing fault-injection discovery.
    # A dashboard job injects e.g. EXP03_freeze_medium_R02; direct CLI runs keep
    # the historical joint_YYYYMMDD_HHMMSS directory format.
    if RUN_NAME_PREFIX:
        import re
        safe_prefix = re.sub(r'[^A-Za-z0-9_-]+', '_', str(RUN_NAME_PREFIX)).strip('_')[:100]
        if not safe_prefix:
            raise ValueError('RUN_NAME_PREFIX has no safe filename characters')
        name = f'joint_{safe_prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    else:
        name = datetime.now().strftime('joint_%Y%m%d_%H%M%S')
    run_dir = ROOT / 'captures' / name
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        'experiment': '10 - joint quality/performance',
        'capture_name_prefix': RUN_NAME_PREFIX,
        'grid_url': rtc.GRID_URL, 'jitsi_url': rtc.JITSI_URL,
        'source_video': CONTAINER_VIDEO, 'reference_video': SOURCE.name,
        'metadata': METADATA.name, 'duration_seconds': args.duration,
        'sample_interval_seconds': args.interval, 'warmup_seconds': 5,
        'stats': 'inbound-rtp via RTCPeerConnection.getStats',
        'capture': 'remote HTMLVideoElement canvas PNG',
        'quality': 'independent 24-bit frame barcode; masked grayscale SSIM and RGB PSNR',
        'time_alignment_note': 'WebRTC snapshot taken immediately BEFORE PNG capture; interval rates cover preceding sample window',
    }
    (run_dir / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Cikti klasoru:', run_dir, flush=True)
    drivers, rows = {}, []
    selected_id, previous, previous_elapsed = None, None, None
    try:
        for name in ('Participant_A', 'Participant_B'):
            print(name, 'baslatiliyor...', flush=True)
            drivers[name] = rtc.start_participant(name)
        b = drivers['Participant_B']
        print('WebRTC alici akisi bekleniyor (en fazla 75 saniye)...', flush=True)
        from selenium.webdriver.support.ui import WebDriverWait

        def is_ready(d):
            data = rtc.fetch_stats(d)
            inbound = rtc.choose_inbound(data)
            video = rtc.choose_remote_video(data)
            return bool(inbound and numeric(inbound.get('framesDecoded')) and
                        video and video.get('readyState', 0) >= 2 and
                        video.get('width', 0) > 0)

        try:
            WebDriverWait(b, 75, poll_frequency=2).until(is_ready)
        except Exception as exc:
            try:
                data = rtc.fetch_stats(b)
                (run_dir / 'not_ready_diagnostics.json').write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            except Exception:
                pass
            raise RuntimeError('WebRTC alici videosu hazir degil; diagnostics dosyasina bakin') from exc
        print('Video hazir. 5 saniye isinma...', flush=True)
        time.sleep(5)
        try:
            ok = b.save_screenshot(str(run_dir / 'Participant_B_ui.png'))
            if not ok:
                print('UYARI: UI ekran goruntusu kaydedilemedi', flush=True)
        except Exception as exc:
            print('UYARI: UI screenshot:', exc, flush=True)
        print(f'{args.duration} saniye / {args.interval} saniyelik araliklarla ortak olcum...', flush=True)
        start = time.monotonic()
        n = args.duration // args.interval + 1
        for index in range(n):
            wait = start + index * args.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            elapsed = time.monotonic() - start
            row = {key: '' for key in FIELDS}
            row.update(sample=index, elapsed_s=round(elapsed, 3),
                       local_time=datetime.now().isoformat(timespec='milliseconds'),
                       status='OK')
            stats_start = time.monotonic()
            try:
                data = rtc.fetch_stats(b)
                (run_dir / f'raw_stats_{index:03d}.json').write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                stream = rtc.choose_inbound(data, selected_id)
                video = rtc.choose_remote_video(data)
                if stream is None:
                    raise RuntimeError('inbound-rtp video statistics not found')
                if selected_id != stream.get('id'):
                    previous, previous_elapsed = None, None
                selected_id = stream['id']
                pcs = data.get('peerConnections', [])
                state = next((p.get('connectionState', '') for p in pcs
                              if p.get('index') == stream.get('pcIndex')), '')
                row.update(inbound_id=selected_id, codec=stream.get('codecMimeType'),
                           connection_state=state, rtp_width=stream.get('frameWidth'),
                           rtp_height=stream.get('frameHeight'),
                           reported_fps=stream.get('framesPerSecond'),
                           frames_decoded=stream.get('framesDecoded'),
                           packets_received=stream.get('packetsReceived'),
                           packets_lost=stream.get('packetsLost'),
                           bytes_received=stream.get('bytesReceived'))
                if numeric(stream.get('jitter')) is not None:
                    row['jitter_ms'] = round(stream['jitter'] * 1000, 3)
                if video:
                    row.update(playback_width=video.get('width'),
                               playback_height=video.get('height'),
                               playback_total_frames=video.get('totalVideoFrames'))
                compute_intervals(row, stream, previous, elapsed, previous_elapsed)
                previous, previous_elapsed = stream, elapsed
            except Exception as exc:
                row['status'] = 'STATS_ERROR'
                row['error'] = str(exc)
                previous, previous_elapsed = None, None
            stats_end = time.monotonic()
            target = run_dir / f'Participant_B_remote_{index:03d}.png'
            try:
                ok, info = capture.capture_frame(b, target)
                row['capture_status'] = 'OK' if ok else 'ERROR'
                row['capture_error'] = '' if ok else info.get('error', 'Unknown')
                if ok:
                    row.update(capture_filename=target.name, capture_width=info['width'],
                               capture_height=info['height'], capture_bytes=info['fileBytes'])
            except Exception as exc:
                row['capture_status'] = 'ERROR'
                row['capture_error'] = repr(exc)
            capture_end = time.monotonic()
            row['capture_offset_ms'] = round((capture_end - stats_start) * 1000, 1)
            row['stats_to_capture_ms'] = round((capture_end - stats_end) * 1000, 1)
            rows.append(row)
            save_csv(run_dir / 'joint_measurements.csv', rows)
            print(f'[{index:02d}] {row["rtp_width"] or "?"}x{row["rtp_height"] or "?"} '
                  f'FPS={rtc.fmt(row["observed_decode_fps"], 1)} '
                  f'Mbps={rtc.fmt(row["bitrate_mbps"], 2)} '
                  f'PNG={row["capture_status"]} '
                  f'{row["capture_width"] or "?"}x{row["capture_height"] or "?"}', flush=True)
    finally:
        for driver in drivers.values():
            try:
                driver.quit()
            except Exception:
                pass

    print('\nVideo yakalama bitti. Referans kareler ile analiz ediliyor...', flush=True)
    reference_info = enrich_quality(rows, run_dir)
    save_csv(run_dir / 'joint_measurements.csv', rows)
    good = [r for r in rows if r['quality_status'] == 'OK']
    groups = {}
    for row in good:
        label = f'{row["capture_width"]}x{row["capture_height"]}'
        groups.setdefault(label, []).append(row)
    grouped = {k: {'n': len(v), 'mean_ssim': avg(v, 'ssim_gray_masked'),
                   'mean_psnr_db': avg(v, 'psnr_rgb_masked_db')}
               for k, v in groups.items()}
    summary = {
        **reference_info, 'samples': len(rows),
        'stats_ok': sum(r['status'] == 'OK' for r in rows),
        'captures_ok': sum(r['capture_status'] == 'OK' for r in rows),
        'quality_ok': len(good),
        'mean_ssim_masked': avg(good, 'ssim_gray_masked'),
        'mean_psnr_masked_db': avg(good, 'psnr_rgb_masked_db'),
        'mean_observed_decode_fps': avg(rows, 'observed_decode_fps'),
        'mean_bitrate_mbps': avg(rows, 'bitrate_mbps'),
        'mean_jitter_ms': avg(rows, 'jitter_ms'),
        'quality_by_received_resolution': grouped,
        'limitations': [
            'Each SSIM value corresponds to the PNG captured immediately after the getStats snapshot.',
            'Bitrate/FPS/loss deltas summarize the preceding interval, not an instantaneous frame property.',
            'Sparse sampling and a short repeating Y4M source; no end-to-end latency estimate.',
            'Resolution and image quality are observational, not causal evidence.',
            'No claim that source 50 FPS equals transmitted FPS.',
        ],
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSONUC - Birlesik deney', flush=True)
    print(f'Kareler: {len(good)}/{len(rows)} senkronize; '
          f'SSIM={rtc.fmt(summary["mean_ssim_masked"], 4)}, '
          f'PSNR={rtc.fmt(summary["mean_psnr_masked_db"], 2)} dB', flush=True)
    print(f'Decode FPS={rtc.fmt(summary["mean_observed_decode_fps"], 2)}; '
          f'bitrate={rtc.fmt(summary["mean_bitrate_mbps"], 2)} Mbps; '
          f'jitter={rtc.fmt(summary["mean_jitter_ms"], 2)} ms', flush=True)
    print('Cozunurluge gore:', grouped, flush=True)
    print('CSV:', run_dir / 'joint_measurements.csv', flush=True)
    print('Ozet:', run_dir / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
