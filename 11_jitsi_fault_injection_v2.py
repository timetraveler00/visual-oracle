"""11 - Controlled SOURCE-SIDE visual fault injection for Jitsi video tests.

Windows Anaconda Prompt (from D:\\visual-oracle-revision):
    python 11_jitsi_fault_injection.py --scenario blur
    python 11_jitsi_fault_injection.py --scenario baseline
    python 11_jitsi_fault_injection.py --scenario color --severity high
    python 11_jitsi_fault_injection.py --all --repeats 3

First, the program creates a deterministic faulty Y4M video from the CLEAN
Shields source. The original 24-cell frame barcode is drawn AFTER the fault,
so receiver-side synchronization remains independent of the quality metric.
It dynamically finds Participant_A's actual Selenium Grid node, copies the
Y4M ONLY to that sender node BEFORE Jitsi navigation, then runs
10_jitsi_joint_quality_performance.py using the faulty video as the camera.
Finally it annotates the joint CSV with the known injected fault ground truth.

Faults are CAMERA-SOURCE CONTENT modifications, NOT simulated packet loss,
WebRTC transport freezes, or network faults. The 'freeze' scenario holds the
image content while the independently generated frame counter advances.

Requires these working scripts in the same folder: 07, 08, 10 (and 05, 09
which 10 imports), plus Docker, Selenium Grid, Jitsi, numpy, OpenCV, Selenium.
No changes to Jitsi / Docker Compose configuration are required.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
SOURCE = ROOT / '720p50_shields_ter.y4m'
NUMBERED = ROOT / '720p50_shields_ter_numbered.y4m'
METADATA = ROOT / '720p50_shields_ter_numbered_metadata.json'
OUTPUT_ROOT = ROOT / 'fault_sources'
CONTAINERS = ('selenium-chrome-1', 'selenium-chrome-2')  # Selenium Grid candidate nodes
SCENARIOS = ('baseline', 'blur', 'color', 'freeze', 'black', 'occlusion')
SEVERITIES = ('low', 'medium', 'high')


def load_module(name, filename):
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f'Gerekli Python dosyasi eksik: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fault_window(scenario, severity, count, fps):
    """Return inclusive frame-index window; other scenarios affect all frames."""
    if scenario == 'freeze':
        start = min(int(3.0 * fps), count - 1)
        duration = {'low': 1.5, 'medium': 4.0, 'high': 6.0}[severity]
        return start, min(count - 1, start + max(1, round(duration * fps)) - 1)
    if scenario == 'black':
        if severity == 'high':
            return 0, count - 1
        fraction = {'low': 0.25, 'medium': 0.5}[severity]
        n = max(1, round(count * fraction))
        start = (count - n) // 2
        return start, start + n - 1
    return 0, count - 1


def is_active(scenario, frame_index, start, end):
    return scenario != 'baseline' and start <= frame_index <= end


def apply_fault(y, u, v, scenario, severity):
    """Manipulate I420 planes without lossy BGR round trips."""
    if scenario == 'blur':
        sigma = {'low': 1.5, 'medium': 3.5, 'high': 6.0}[severity]
        y[:] = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma)
        u[:] = cv2.GaussianBlur(u, (0, 0), sigmaX=max(0.8, sigma / 2))
        v[:] = cv2.GaussianBlur(v, (0, 0), sigmaX=max(0.8, sigma / 2))
    elif scenario == 'color':
        # Chroma-only fault: useful to test grayscale SSIM blind spots.
        shift = {'low': 15, 'medium': 35, 'high': 55}[severity]
        u[:] = np.clip(u.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        v[:] = np.clip(v.astype(np.int16) - shift, 0, 255).astype(np.uint8)
    elif scenario == 'black':
        y[:] = 16
        u[:] = 128
        v[:] = 128
    elif scenario == 'occlusion':
        fraction = {'low': 0.15, 'medium': 0.30, 'high': 0.50}[severity]
        h, w = y.shape
        # Centered square-ish rectangle with target fraction of frame area.
        rw = (round(w * math.sqrt(fraction)) // 2) * 2
        rh = (round(h * math.sqrt(fraction)) // 2) * 2
        x0, y0 = ((w - rw) // 4) * 2, ((h - rh) // 4) * 2
        x1, y1 = x0 + rw, y0 + rh
        y[y0:y1, x0:x1] = 16
        u[y0//2:y1//2, x0//2:x1//2] = 128
        v[y0//2:y1//2, x0//2:x1//2] = 128
    elif scenario in ('freeze', 'baseline'):
        # Freeze is applied by selecting a previous raw frame, not here.
        pass
    else:
        raise ValueError('Bilinmeyen hata senaryosu: ' + scenario)


def prepare_video(scenario, severity, quality, overlay, regenerate=False):
    metadata = json.loads(METADATA.read_text(encoding='utf-8-sig'))
    w, h, fps, offsets = quality.index_y4m(SOURCE)
    count = len(offsets)
    if (w, h, count) != (int(metadata['width']), int(metadata['height']),
                          int(metadata['frame_count'])):
        raise ValueError('Orijinal Y4M ve numbered metadata uyusmuyor')
    start, end = fault_window(scenario, severity, count, fps)
    if scenario == 'baseline':
        if not NUMBERED.is_file():
            raise FileNotFoundError('Numbered source not found: ' + str(NUMBERED))
        return NUMBERED, metadata, (start, end), {'scenario': scenario,
            'severity': 'none', 'frame_count': count, 'fps': fps,
            'fault_start_frame': None, 'fault_end_frame': None,
            'source_video': NUMBERED.name, 'source_sha256': sha256(NUMBERED)}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f'720p50_shields_ter_fault_{scenario}_{severity}.y4m'
    info_path = path.with_suffix('.json')
    expected = {'scenario': scenario, 'severity': severity, 'frame_count': count,
                'fps': fps, 'fault_start_frame': start, 'fault_end_frame': end,
                'reference_video': SOURCE.name, 'source_video': path.name,
                'overlay': 'original 24-bit barcode redrawn AFTER fault'}
    if path.exists() and not regenerate:
        if not info_path.exists():
            raise RuntimeError(f'Var olan Y4M icin metadata yok: {info_path}. --regenerate kullanin.')
        old = json.loads(info_path.read_text(encoding='utf-8-sig'))
        if any(old.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Var olan Y4M parametreleri uyusmuyor; --regenerate kullanin.')
        if path.stat().st_size < SOURCE.stat().st_size * 0.95:
            raise RuntimeError('Var olan Y4M eksik olabilir; --regenerate kullanin.')
        print('Mevcut hata videosu yeniden kullaniliyor:', path, flush=True)
        return path, metadata, (start, end), old
    if regenerate:
        path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)

    if shutil.disk_usage(OUTPUT_ROOT).free < SOURCE.stat().st_size * 1.2:
        raise OSError('Hata videosu icin yeterli bos disk alani yok')
    tmp = path.with_suffix('.y4m.part')
    tmp.unlink(missing_ok=True)
    frame_size = w * h * 3 // 2
    y_size, uv_size = w * h, w * h // 4
    freeze_anchor = None
    preview_written = False
    preview_path = path.with_name(path.stem + '_preview.png')
    print(f'Hata videosu uretiliyor: {scenario}/{severity}; FRAME {start}..{end}', flush=True)
    try:
        with SOURCE.open('rb') as src, tmp.open('xb') as dst:
            header = src.readline()
            quality.parse_header(header)  # validate Y4M format
            dst.write(header)
            for index in range(count):
                marker = src.readline()
                if not marker.startswith(b'FRAME'):
                    raise ValueError(f'FRAME marker not found: {index}')
                raw = src.read(frame_size)
                if len(raw) != frame_size:
                    raise ValueError(f'Kaynak kare eksik: {index}')
                active = is_active(scenario, index, start, end)
                if scenario == 'freeze' and index == start:
                    freeze_anchor = raw
                if scenario == 'freeze' and active:
                    raw = freeze_anchor
                buf = bytearray(raw)
                y = np.frombuffer(buf, np.uint8, count=y_size).reshape(h, w)
                u = np.frombuffer(buf, np.uint8, count=uv_size, offset=y_size).reshape(h//2, w//2)
                v = np.frombuffer(buf, np.uint8, count=uv_size, offset=y_size+uv_size).reshape(h//2, w//2)
                if active and scenario != 'freeze':
                    apply_fault(y, u, v, scenario, severity)
                # Preserve frame identity even for black/frozen video CONTENT.
                overlay.draw_overlay(y, u, v, index, fps, w, h)
                dst.write(marker)
                dst.write(buf)
                if active and not preview_written:
                    arr = np.frombuffer(buf, np.uint8).reshape(h*3//2, w)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)
                    if not cv2.imwrite(str(preview_path), bgr):
                        raise OSError('Onizleme yazilamadi')
                    preview_written = True
                if (index + 1) % 150 == 0:
                    print(f'  {index + 1}/{count} kare', flush=True)
            if src.read(1):
                raise ValueError('Kaynakta beklenenden fazla kare var')
        tmp.replace(path)
        expected['source_sha256'] = sha256(path)
        info_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f'Hazir: {path} ({path.stat().st_size/1024**2:.1f} MiB)', flush=True)
    print('Onizleme:', preview_path, flush=True)
    return path, metadata, (start, end), expected


def grid_session_node_uri(grid_url, session_id):
    """Get the node URI assigned to this *specific* Selenium Grid session."""
    url = grid_url.rstrip('/') + '/graphql'
    # Selenium Grid 4 GraphQL: session(id) includes nodeUri and nodeId.
    query = ('query ($id: String!) {'
             ' session(id: $id) { id nodeId nodeUri } }')
    payload = json.dumps({'query': query, 'variables': {'id': session_id}}).encode('utf-8')
    request = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'},
                                     method='POST')
    with urllib.request.urlopen(request, timeout=12) as response:
        result = json.load(response)
    if result.get('errors'):
        raise RuntimeError(f'Grid GraphQL sorgu hatasi: {result["errors"]}')
    session = (result.get('data') or {}).get('session')
    if not session or session.get('id') != session_id or not session.get('nodeUri'):
        raise RuntimeError(f'Grid session or nodeUri not found: {session_id}; {session}')
    return session['nodeUri']


def container_node_addresses(container):
    """Get all Docker network IPs, so no Grid-node ordering is assumed."""
    process = subprocess.run(
        ['docker', 'inspect', '--format', '{{json .NetworkSettings.Networks}}', container],
        check=True, capture_output=True, text=True, encoding='utf-8')
    networks = json.loads(process.stdout)
    addresses = set()
    for config in networks.values():
        for key in ('IPAddress', 'GlobalIPv6Address'):
            address = config.get(key)
            if address:
                addresses.add(address.strip('[]').lower())
    addresses.add(container.lower())
    return addresses


def resolve_sender_container(node_uri):
    host = urllib.parse.urlsplit(node_uri).hostname
    if not host:
        raise RuntimeError(f'Grid nodeUri gecersiz: {node_uri}')
    host = host.lower()
    matches = []
    diagnostic = {}
    for container in CONTAINERS:
        addresses = container_node_addresses(container)
        diagnostic[container] = sorted(addresses)
        if host in addresses:
            matches.append(container)
    if len(matches) != 1:
        raise RuntimeError(
            'Gonderici Selenium node Docker konteynerine guvenle eslenemedi. '
            f'nodeUri={node_uri}, Docker IP adresleri={diagnostic}. '
            'Yanlis kamerayi kullanmamak icin test durduruldu. '
            'Gerekirse --copy-both ile eski davranisi kullanabilirsiniz.')
    return matches[0]


def docker_copy_to_container(path, container):
    print(f'Docker kopya (yalnizca gonderici): {container}', flush=True)
    subprocess.run(['docker', 'cp', str(path), f'{container}:/tmp/{path.name}'], check=True)
    subprocess.run(['docker', 'exec', container, 'test', '-s', f'/tmp/{path.name}'], check=True)


def docker_copy_both(path):
    """Explicit compatibility fallback, never the default."""
    for container in CONTAINERS:
        print(f'Docker kopya (--copy-both): {container}', flush=True)
        subprocess.run(['docker', 'cp', str(path), f'{container}:/tmp/{path.name}'], check=True)
        subprocess.run(['docker', 'exec', container, 'test', '-s', f'/tmp/{path.name}'], check=True)


def sender_aware_participant_starter(rtc, video_path):
    """Stage sender video in the allocated Grid node, then use shared join flow."""
    def start_participant(name):
        if name != 'Participant_A':
            return rtc.start_participant_original(name)
        options = rtc.Options()
        for flag in (
            '--ignore-certificate-errors',
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
            '--autoplay-policy=no-user-gesture-required',
            '--window-size=1280,720',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
        ):
            options.add_argument(flag)
        options.add_argument(f'--use-file-for-fake-video-capture=/tmp/{video_path.name}')
        options.page_load_strategy = 'eager'
        driver = rtc.webdriver.Remote(command_executor=rtc.GRID_URL, options=options)
        try:
            node_uri = grid_session_node_uri(rtc.GRID_URL, driver.session_id)
            container = resolve_sender_container(node_uri)
            print(f'Participant_A Grid node: {node_uri} -> {container}', flush=True)
            docker_copy_to_container(video_path, container)
            driver.set_script_timeout(30)
            # Install hook before navigating to Jitsi.
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument',
                                   {'source': rtc.PC_HOOK_JS})
            driver.get(rtc.JITSI_URL)
            rtc.join_meeting(driver, name)
            return driver
        except Exception:
            try:
                driver.quit()
            except Exception:
                pass
            raise
    return start_participant


def read_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def average(rows, key):
    values = []
    for row in rows:
        try:
            val = float(row[key])
            if math.isfinite(val):
                values.append(val)
        except (KeyError, ValueError, TypeError):
            pass
    return round(sum(values) / len(values), 6) if values else None


def annotate_run(run_dir, scenario, severity, window, info, quality, metadata):
    rows = read_csv(run_dir / 'joint_measurements.csv')
    start, end = window
    w, h, fps, offsets = quality.index_y4m(SOURCE)
    with SOURCE.open('rb') as ref_file:
        for row in rows:
            row['scenario'] = scenario
            row['severity'] = 'none' if scenario == 'baseline' else severity
            row['fault_active'] = ''
            row['mae_rgb_masked'] = ''
            row['delta_e76_lab_masked'] = ''
            if row.get('quality_status') != 'OK':
                continue
            index = int(row['frame_index'])
            active = is_active(scenario, index, start, end)
            row['fault_active'] = int(active)
            received = cv2.imread(str(run_dir / row['capture_filename']), cv2.IMREAD_COLOR)
            if received is None:
                continue
            rh, rw = received.shape[:2]
            reference = quality.read_reference(ref_file, offsets[index], w, h, rw, rh)
            mask, _ = quality.valid_pixel_mask(received.shape, metadata)
            row['mae_rgb_masked'] = round(float(np.abs(
                reference.astype(np.float32) - received.astype(np.float32))[mask].mean()), 5)
            # OpenCV float32 LAB uses L* 0..100 and a*, b* approx -128..127.
            ref_lab = cv2.cvtColor(reference.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            rec_lab = cv2.cvtColor(received.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            delta = np.linalg.norm(ref_lab - rec_lab, axis=2)
            row['delta_e76_lab_masked'] = round(float(delta[mask].mean()), 5)
    output = run_dir / 'fault_measurements.csv'
    fields = list(rows[0]) if rows else []
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    good = [r for r in rows if r.get('quality_status') == 'OK']
    active = [r for r in good if r['fault_active'] == 1]
    inactive = [r for r in good if r['fault_active'] == 0]
    result = {
        'scenario': scenario, 'severity': 'none' if scenario == 'baseline' else severity,
        'run_folder': str(run_dir), 'sample_count': len(rows), 'quality_ok': len(good),
        'active_samples': len(active), 'inactive_samples': len(inactive),
        'mean_ssim_all': average(good, 'ssim_gray_masked'),
        'mean_ssim_fault_active': average(active, 'ssim_gray_masked'),
        'mean_psnr_fault_active_db': average(active, 'psnr_rgb_masked_db'),
        'mean_delta_e76_fault_active': average(active, 'delta_e76_lab_masked'),
        'mean_ssim_fault_inactive': average(inactive, 'ssim_gray_masked'),
        'mean_decode_fps': average(rows, 'observed_decode_fps'),
        'mean_bitrate_mbps': average(rows, 'bitrate_mbps'),
        'fault_start_frame': None if scenario == 'baseline' else start,
        'fault_end_frame': None if scenario == 'baseline' else end,
        'video_sha256': info['source_sha256'],
        'methodology': 'Fault applied to camera-source Y4M before Jitsi; frame barcode drawn AFTER fault; receiver compared to CLEAN original source.',
        'limitations': [
            'Not a WebRTC network fault; no synthetic packet loss or transport freeze.',
            'Source video repeats approximately every 10.08 seconds.',
            'Sparse 5-second samples; freeze/black active windows may be missed.',
            'Single run does not establish detection accuracy, thresholds, or statistical significance.',
            'Frame labels are ground truth for injected content, not an independent time-of-arrival measurement.',
        ],
    }
    (run_dir / 'fault_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    settings_path = run_dir / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    settings['fault_injection'] = {**info, 'fault_window': [start, end]}
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    summary_path = ROOT / 'captures' / 'fault_runs_summary.csv'
    cols = ['scenario', 'severity', 'run_folder', 'sample_count', 'quality_ok',
            'active_samples', 'inactive_samples', 'mean_ssim_all',
            'mean_ssim_fault_active', 'mean_psnr_fault_active_db',
            'mean_delta_e76_fault_active', 'mean_ssim_fault_inactive',
            'mean_decode_fps', 'mean_bitrate_mbps', 'fault_start_frame',
            'fault_end_frame', 'video_sha256']
    with summary_path.open('a', newline='', encoding='utf-8-sig' if not summary_path.exists() else 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(result)
    print('\n11 - HATA ENJEKSIYONU SONUCU', flush=True)
    print(f'Senaryo: {scenario}/{severity} | Senkronize: {len(good)}/{len(rows)}', flush=True)
    print(f'Hata aktif ornek: {len(active)} | Inaktif: {len(inactive)}', flush=True)
    print(f'SSIM (aktif): {result["mean_ssim_fault_active"]} | '
          f'PSNR (aktif): {result["mean_psnr_fault_active_db"]} dB | '
          f'DeltaE76 (aktif): {result["mean_delta_e76_fault_active"]}', flush=True)
    print('Ayrintili CSV:', output, flush=True)
    print('Tum kosularin ozeti:', summary_path, flush=True)
    return result


def run_once(scenario, args, quality, overlay):
    path, metadata, window, info = prepare_video(scenario, args.severity,
                                                  quality, overlay, args.regenerate)
    if args.prepare_only:
        print('Sadece hazirlama yapildi; Jitsi testi baslatilmadi.', flush=True)
        return None
    # Load existing proven joint test only when actually running Jitsi.
    joint = load_module('jitsi_joint10_for_faults', '10_jitsi_joint_quality_performance.py')
    joint.CONTAINER_VIDEO = f'/tmp/{path.name}'
    if args.copy_both:
        docker_copy_both(path)
    else:
        # Only Participant_A gets a source file. Participant_B is unchanged.
        joint.rtc.start_participant_original = joint.rtc.start_participant
        joint.rtc.start_participant = sender_aware_participant_starter(joint.rtc, path)
    before = set((ROOT / 'captures').glob('joint_*'))
    original_argv = sys.argv
    try:
        sys.argv = [str(ROOT / '10_jitsi_joint_quality_performance.py'),
                    '--duration', str(args.duration), '--interval', str(args.interval)]
        print(f'\nJitsi testi basliyor: {scenario}/{args.severity}', flush=True)
        joint.main()
    finally:
        sys.argv = original_argv
    new_dirs = sorted(set((ROOT / 'captures').glob('joint_*')) - before)
    if len(new_dirs) != 1:
        raise RuntimeError(f'Yeni joint klasoru kesin belirlenemedi: {new_dirs}')
    return annotate_run(new_dirs[0], scenario, args.severity, window, info,
                        quality, metadata)


def main():
    parser = argparse.ArgumentParser(description='Controlled sender-side visual faults for Jitsi')
    parser.add_argument('--scenario', choices=SCENARIOS, default='blur',
                        help='Varsayilan: blur (ilk dogrulama icin)')
    parser.add_argument('--severity', choices=SEVERITIES, default='medium')
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--all', action='store_true', help='6 senaryoyu sirayla calistir')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Sadece Y4M uret; Docker/Jitsi calistirma')
    parser.add_argument('--copy-both', action='store_true',
                        help='Grid node belirleme desteklenmiyorsa eski iki node kopyalama')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate existing faulty Y4M file')
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0 or args.duration % args.interval:
        parser.error('--duration pozitif olmali ve --interval degerine tam bolunmeli')
    if args.repeats < 1:
        parser.error('--repeats >= 1 olmali')
    for p in (SOURCE, METADATA):
        if not p.is_file():
            raise FileNotFoundError(f'Required file not found: {p}')
    quality = load_module('jitsi_quality08_faults', '08_jitsi_synchronized_quality.py')
    overlay = load_module('jitsi_overlay07_faults', '07_create_numbered_y4m.py')
    scenarios = SCENARIOS if args.all else (args.scenario,)
    for s in scenarios:
        for rep in range(args.repeats):
            if args.all or args.repeats > 1:
                print(f'\n=== {s.upper()} / tekrar {rep+1}/{args.repeats} ===', flush=True)
            run_once(s, args, quality, overlay)
            if not args.prepare_only and (s != scenarios[-1] or rep < args.repeats - 1):
                print('Oturumlar arasi 10 sn bekleniyor...', flush=True)
                time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('\nHATA:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
