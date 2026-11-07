"""Visual Oracle dashboard job runner. Runs existing 07/08/10/11 code unchanged.

Usage (normally launched by 12_visual_oracle_dashboard.py):
    python 12_visual_oracle_runner.py --job captures/dashboard_jobs/job_x

Source-side faults only. Exactly two participants per run. Runs sequentially.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
import importlib
import importlib.util
from importlib import import_module

english_logs = import_module('12_english_logs')
english_logs.install()
perceptual_module = import_module('12_perceptual_metrics')
JOBS = ROOT / 'captures' / 'dashboard_jobs'
VIDEO_EXTENSIONS = {'.y4m', '.mp4', '.mkv', '.mov', '.avi', '.webm'}
FAULTS = {'baseline', 'blur', 'color', 'freeze', 'black', 'occlusion'}
SEVERITIES = {'low', 'medium', 'high'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def available_nodes():
    proc = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'],
                          capture_output=True, text=True, encoding='utf-8', timeout=20)
    if proc.returncode:
        raise RuntimeError('Docker is unavailable: ' + proc.stderr.strip())
    return sorted((name.strip() for name in proc.stdout.splitlines()
                   if re.fullmatch(r'selenium-chrome-\d+', name.strip())),
                  key=lambda x: int(x.rsplit('-', 1)[1]))


def resolve_video(relative_path):
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError('No video selected')
    path = (ROOT / relative_path).resolve()
    allowed_roots = [(ROOT / 'videos').resolve()]
    if path.parent not in allowed_roots:
        raise ValueError('Videos must be located directly inside the videos folder')
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f'Video not found or unsupported format: {path}')
    if '_numbered' in path.stem or '_fault_' in path.stem:
        raise ValueError('Select an original video (without frame numbering or injected faults)')
    return path


def validate(config):
    if not isinstance(config, dict):
        raise ValueError('Scenario must be a JSON object')
    video = resolve_video(config.get('video'))
    node_count = int(config.get('node_count', 2))
    duration = int(config.get('duration', 60))
    interval = int(config.get('interval', 5))
    repeats = int(config.get('repeats', 1))
    clip_seconds = int(config.get('clip_seconds', 10))
    if not 2 <= node_count <= 10:
        raise ValueError('Node count must be between 2 and 10')
    if not 5 <= duration <= 3600 or not 1 <= interval <= 60 or duration % interval:
        raise ValueError('Duration must be 5–3600 seconds and divisible by the sampling interval')
    if not 1 <= repeats <= 20:
        raise ValueError('Repeat count must be between 1 and 20')
    if not 3 <= clip_seconds <= 60:
        raise ValueError('Clip duration must be between 3 and 60 seconds')
    conditions = config.get('conditions', [])
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 16:
        raise ValueError('Select between 1 and 16 conditions')
    cleaned = []
    for item in conditions:
        if not isinstance(item, dict):
            raise ValueError('Invalid condition format')
        scenario = item.get('scenario')
        severity = item.get('severity', 'medium')
        if scenario not in FAULTS or severity not in SEVERITIES:
            raise ValueError(f'Invalid fault type/severity: {item}')
        cleaned.append({'scenario': scenario, 'severity': severity})
    metrics = config.get('perceptual_metrics', [])
    if not isinstance(metrics, list) or any(m not in perceptual_module.METRIC_COLUMNS for m in metrics):
        raise ValueError('Invalid perceptual metric selection')
    return {'perceptual_metrics': list(dict.fromkeys(metrics)), 'video': video, 'node_count': node_count, 'duration': duration,
            'interval': interval, 'repeats': repeats, 'clip_seconds': clip_seconds,
            'conditions': cleaned}


def convert_video_if_needed(video, clip_seconds):
    """Return 8-bit 720p YUV420 Y4M. Lossy input is normalized with FFmpeg."""
    if video.suffix.lower() == '.y4m':
        return video
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg must be available on PATH to convert MP4/MKV/MOV; otherwise select a Y4M file')
    cache = ROOT / 'prepared_videos'
    cache.mkdir(exist_ok=True)
    stamp = f'{video.resolve()}|{video.stat().st_size}|{video.stat().st_mtime_ns}|{clip_seconds}|1280x720|30'
    digest = hashlib.sha256(stamp.encode('utf-8')).hexdigest()[:12]
    stem = re.sub(r'[^a-zA-Z0-9_-]', '_', video.stem)[:36]
    dest = cache / f'{stem}_{digest}.y4m'
    if dest.is_file() and dest.stat().st_size > 1024 * 1024:
        print('Donusturulmus video onbellekten:', dest, flush=True)
        return dest
    part = cache / f'{stem}_{digest}.partial.y4m'
    part.unlink(missing_ok=True)
    # Enforce fixed geometry for the existing 24-cell frame-barcode scheme.
    vf = 'fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p'
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
           '-vf', vf, '-frames:v', str(clip_seconds * 30), '-an',
           '-f', 'yuv4mpegpipe', str(part)]
    print('FFmpeg ile 720p/30FPS Y4M hazirlaniyor...', flush=True)
    try:
        subprocess.run(cmd, check=True)
        if part.stat().st_size < 1024 * 1024:
            raise RuntimeError('FFmpeg generated an empty or unexpectedly short Y4M file')
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def ensure_numbered(source):
    output = source.with_name(source.stem + '_numbered.y4m')
    meta = output.with_name(output.stem + '_metadata.json')
    # Metadata is tied to source path, size and mtime, not just a filename.
    stamp_path = output.with_name(output.stem + '_source_fingerprint.json')
    fingerprint = {'path': str(source.resolve()), 'size': source.stat().st_size,
                   'mtime_ns': source.stat().st_mtime_ns}
    if output.is_file() and meta.is_file() and stamp_path.is_file():
        if json.loads(stamp_path.read_text(encoding='utf-8')) == fingerprint:
            print('Reusing cached numbered Y4M:', output, flush=True)
            return output, meta
    # For previously generated originals (07), reuse existing metadata safely
    # if metadata explicitly points to the selected source and frame count matches.
    if output.is_file() and meta.is_file() and not stamp_path.is_file():
        existing = json.loads(meta.read_text(encoding='utf-8-sig'))
        if Path(existing.get('source', '')).resolve() == source.resolve():
            stamp_path.write_text(json.dumps(fingerprint), encoding='utf-8')
            print('Reusing existing numbered video:', output, flush=True)
            return output, meta
    if output.exists():
        # Do not destroy an existing source file; preserve it if invalid.
        backup = output.with_name(output.name + '.old')
        if backup.exists():
            raise RuntimeError('An old numbered Y4M backup already exists; inspect it manually: ' + str(backup))
        output.rename(backup)
        if meta.exists():
            meta.rename(meta.with_name(meta.name + '.old'))
    overlay = import_file('visual_overlay07', ROOT / '07_create_numbered_y4m.py')
    overlay.generate(source, output)
    stamp_path.write_text(json.dumps(fingerprint), encoding='utf-8')
    return output, meta


def run_one(source, numbered, metadata, condition, duration, interval, node_names):
    fault = import_file('visual_fault11_dashboard', ROOT / '11_jitsi_fault_injection_v2.py')
    fault.SOURCE = source
    fault.NUMBERED = numbered
    fault.METADATA = metadata
    source_stamp = f'{source.resolve()}|{source.stat().st_size}|{source.stat().st_mtime_ns}'
    source_key = hashlib.sha256(source_stamp.encode('utf-8')).hexdigest()[:10]
    fault.OUTPUT_ROOT = ROOT / 'fault_sources' / (source.stem + '_' + source_key)
    fault.CONTAINERS = tuple(node_names)
    original_loader = fault.load_module

    def patched_loader(name, filename):
        module = original_loader(name, filename)
        if filename == '10_jitsi_joint_quality_performance.py':
            module.SOURCE = source
            module.METADATA = metadata
            module.NUMBERED = numbered.name
            module.CONTAINER_VIDEO = '/tmp/' + numbered.name
        return module

    fault.load_module = patched_loader
    quality = original_loader('visual_quality08_dashboard', '08_jitsi_synchronized_quality.py')
    overlay = original_loader('visual_overlay07_dashboard', '07_create_numbered_y4m.py')
    from types import SimpleNamespace
    args = SimpleNamespace(severity=condition['severity'], regenerate=False,
                           prepare_only=False, copy_both=False,
                           duration=duration, interval=interval)
    return fault.run_once(condition['scenario'], args, quality, overlay)


def make_summary_csv(job_dir, results):
    columns = ['scenario', 'severity', 'run_folder', 'quality_ok', 'sample_count',
               'active_samples', 'mean_ssim_all', 'mean_ssim_fault_active',
               'mean_psnr_fault_active_db', 'mean_delta_e76_fault_active',
               'mean_decode_fps', 'mean_bitrate_mbps',
               'mean_dists_similarity_all', 'mean_dinov2_cosine_all', 'mean_dinov2_large_cosine_all', 'mean_openclip_cosine_all',
               'mean_dists_similarity_fault_active', 'mean_dinov2_cosine_fault_active', 'mean_dinov2_large_cosine_fault_active',
               'mean_openclip_cosine_fault_active']
    with (job_dir / 'job_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def run(job_dir):
    config = json.loads((job_dir / 'scenario.json').read_text(encoding='utf-8'))
    status_path = job_dir / 'status.json'
    state = {'state': 'starting', 'started_at': now(), 'updated_at': now(),
             'pid': os.getpid(), 'current': 0, 'total': 0, 'results': [],
             'message': 'Preparing'}

    def update(**kwargs):
        state.update(kwargs)
        state['updated_at'] = now()
        atomic_json(status_path, state)

    update()
    try:
        cfg = validate(config)
        nodes = available_nodes()
        if len(nodes) < cfg['node_count']:
            raise RuntimeError(f'{cfg["node_count"]} nodes requested, but only {len(nodes)} are running: {nodes}. '
                               'Start the node containers before running the experiment.')
        for name in ('07_create_numbered_y4m.py', '08_jitsi_synchronized_quality.py',
                     '05_jitsi_video_capture.py', '09_jitsi_webrtc_stats.py',
                     '10_jitsi_joint_quality_performance.py', '11_jitsi_fault_injection_v2.py'):
            if not (ROOT / name).is_file():
                raise FileNotFoundError(f'Required existing script is missing: {name}')
        total = len(cfg['conditions']) * cfg['repeats']
        update(state='preparing', total=total, message='Preparing source video')
        # Fail before opening Jitsi if optional model packages/weights are unavailable.
        engine = perceptual_module.PerceptualEngine(cfg['perceptual_metrics'])
        source = convert_video_if_needed(cfg['video'], cfg['clip_seconds'])
        numbered, metadata = ensure_numbered(source)
        update(state='running', message='Starting Jitsi tests',
               prepared_source=str(source), numbered_source=str(numbered))
        for cond in cfg['conditions']:
            for rep in range(cfg['repeats']):
                if (job_dir / 'cancel.flag').exists():
                    update(state='cancelled', message='Cancelled by user')
                    return
                label = f'{cond["scenario"]}/{cond["severity"]} repeat {rep+1}/{cfg["repeats"]}'
                update(message=label)
                print('\n' + '=' * 70 + '\n' + label + '\n' + '=' * 70, flush=True)
                result = run_one(source, numbered, metadata, cond, cfg['duration'], cfg['interval'], nodes)
                if result is None:
                    raise RuntimeError('Test finished but no result was returned')
                if engine.selected:
                    update(message=f'Computing perceptual metrics: {label}')
                    extra = perceptual_module.enrich_run(
                        result['run_folder'], source, metadata, engine, ROOT)
                    result.update(extra)
                result['repeat_index'] = rep + 1
                state['results'].append(result)
                update(current=state['current'] + 1, message=label + ' completed')
                make_summary_csv(job_dir, state['results'])
                if state['current'] < total:
                    print('Waiting 10 seconds before the next run...', flush=True)
                    time.sleep(10)
        update(state='completed', finished_at=now(), message='All experiment runs completed')
    except Exception as exc:
        traceback.print_exc()
        update(state='failed', finished_at=now(), message=str(exc),
               error_traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job_dir = Path(args.job).resolve()
    if job_dir.parent != JOBS.resolve() or not job_dir.is_dir():
        parser.error('Invalid job directory')
    run(job_dir)


if __name__ == '__main__':
    main()
