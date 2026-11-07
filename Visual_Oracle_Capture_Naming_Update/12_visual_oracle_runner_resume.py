"""Visual Oracle dashboard job runner. Runs existing 07/08/10/11 code unchanged.

Usage (normally launched by 12_visual_oracle_dashboard.py):
    python 12_visual_oracle_runner.py --job captures/dashboard_jobs/job_x

Resume a failed job without rerunning saved results:
    python 12_visual_oracle_runner.py --job captures/dashboard_jobs/job_x --resume

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
    """Windows-safe best-effort atomic status update with recovery snapshot.

    Windows readers, sync clients and antivirus can briefly deny os.replace.
    Never turn a completed experiment into a failed job just because a
    progress/status file was locked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    recovery = path.with_name('status_recovery.json')
    for attempt in range(25):
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.{attempt}.tmp')
        try:
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, path)
            try:
                recovery.unlink(missing_ok=True)
            except OSError:
                pass
            return
        except PermissionError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.12)
    # Keep the completed results recoverable without falsely failing the job.
    recovery.write_text(payload, encoding='utf-8')
    print('[WARNING] status.json was locked; wrote status_recovery.json instead.', flush=True)


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


def capture_name_prefix(experiment_name, condition, repeat):
    """Readable, Windows-safe label; actual folder gets a unique timestamp."""
    match = re.search(r'EXP[0-9]+', str(experiment_name), re.IGNORECASE)
    if match:
        experiment = match.group(0).upper()
    else:
        experiment = re.sub(r'[^A-Za-z0-9_-]+', '_', str(experiment_name)).strip('_')[:40] or 'Experiment'
    scenario = condition['scenario']
    severity = 'none' if scenario == 'baseline' else condition['severity']
    return f'{experiment}_{scenario}_{severity}_R{repeat:02d}'


def write_capture_info(result, job_dir, experiment_name, condition, repeat, plan_index):
    """Optional human-readable sidecar. Never fail an experiment on an info file."""
    try:
        folder = Path(result['run_folder'])
        info = folder / 'CAPTURE_INFO.txt'
        if info.exists():
            return
        content = (f'VISUAL_ORACLE_CAPTURE_INFO_V1\n'
                   f'Experiment : {experiment_name}\n'
                   f'Job        : {job_dir.name}\n'
                   f'Condition  : {condition["scenario"]}/{"none" if condition["scenario"] == "baseline" else condition["severity"]}\n'
                   f'Plan index : {plan_index}\n'
                   f'Repeat     : {repeat}\n'
                   f'Original   : {folder}\n')
        info.write_text(content, encoding='utf-8')
    except OSError as exc:
        print(f'[WARNING] Could not write CAPTURE_INFO.txt: {exc}', flush=True)


def run_one(source, numbered, metadata, condition, duration, interval, node_names, capture_prefix=None):
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
            module.RUN_NAME_PREFIX = capture_prefix or ''
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
    columns = ['plan_index', 'repeat_index', 'scenario', 'severity', 'run_folder', 'run_valid',
               'stalled_intervals', 'validity_warning', 'quality_ok', 'sample_count',
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


def check_stream_continuity(run_folder):
    """Use corroborated RTP+playback liveness, not stale RTP IDs alone."""
    path = Path(run_folder) / 'joint_measurements.csv'
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    stalled = [str(r.get('sample', '?')) for r in rows if r.get('stream_liveness') == 'STALLED']
    longest, current = 0, 0
    for row in rows:
        current = current + 1 if row.get('stream_liveness') == 'STALLED' else 0
        longest = max(longest, current)
    valid = longest < 2
    reason = (f'Corroborated playback stall: {longest} consecutive intervals '
              f'(samples {", ".join(stalled)})' if not valid else '')
    return valid, reason, len(stalled)



def plan_runs(cfg):
    """Stable, zero-based plan order; one entry per independent repetition."""
    return [(cond, rep + 1) for cond in cfg['conditions']
            for rep in range(cfg['repeats'])]


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def read_saved_state(job_dir):
    """Prefer the most complete valid status snapshot (including recovery)."""
    candidates = []
    for path in (job_dir / 'status.json', job_dir / 'status_recovery.json'):
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text(encoding='utf-8-sig'))
            if isinstance(obj, dict) and isinstance(obj.get('results', []), list):
                candidates.append((len(obj.get('results', [])),
                                   str(obj.get('updated_at', '')), obj, path))
        except (OSError, ValueError) as exc:
            print(f'[WARNING] Cannot read {path.name}: {exc}', flush=True)
    if not candidates:
        raise RuntimeError('No readable status.json/status_recovery.json; refusing unsafe resume')
    _, _, state, path = max(candidates, key=lambda x: (x[0], x[1]))
    print(f'Resuming from {path.name}', flush=True)
    return state


def verify_completed_results(results, plan, job_dir):
    """Fail closed on unexpected or missing historical data; never silently rerun it."""
    if len(results) > len(plan):
        raise RuntimeError(f'Saved results ({len(results)}) exceed plan ({len(plan)})')
    folders = set()
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise RuntimeError(f'Invalid saved result at position {i + 1}')
        cond, repeat = plan[i]
        if (result.get('scenario') != cond['scenario'] or
                result.get('severity') != cond['severity']):
            raise RuntimeError(f'Plan mismatch at run {i + 1}: saved '
                               f'{result.get("scenario")}/{result.get("severity")}, expected '
                               f'{cond["scenario"]}/{cond["severity"]}. '
                               'Do not modify scenario.json for a resumed job.')
        if 'repeat_index' in result and int(result['repeat_index']) != repeat:
            raise RuntimeError(f'Repeat mismatch at saved run {i + 1}')
        if 'plan_index' in result and int(result['plan_index']) != i + 1:
            raise RuntimeError(f'Plan index mismatch at saved run {i + 1}')
        folder_text = result.get('run_folder')
        if not isinstance(folder_text, str) or not folder_text.strip():
            raise RuntimeError(f'Run {i + 1} has no run_folder')
        folder = Path(folder_text)
        if not folder.is_absolute():
            folder = (ROOT / folder).resolve()
        if str(folder).lower() in folders:
            raise RuntimeError(f'Duplicate output folder in saved results: {folder}')
        folders.add(str(folder).lower())
        if not (folder / 'joint_measurements.csv').is_file():
            raise RuntimeError(f'Missing saved measurements for run {i + 1}: {folder}. '
                               'Stopping to avoid duplicating or losing measurements.')
        result['plan_index'] = i + 1
        result['repeat_index'] = repeat
    return len(results)


def run(job_dir, resume=False):
    config = json.loads((job_dir / 'scenario.json').read_text(encoding='utf-8'))
    status_path = job_dir / 'status.json'
    digest = config_digest(config)
    cfg = validate(config)
    plan = plan_runs(cfg)
    total = len(plan)

    if resume:
        state = read_saved_state(job_dir)
        if state.get('state') in ('running', 'preparing', 'starting'):
            raise RuntimeError('Job status is still active. Verify the original runner is stopped '
                               'before using --resume; refusing concurrent runs.')
        if state.get('config_sha256') and state['config_sha256'] != digest:
            raise RuntimeError('scenario.json changed since the previous run; refusing resume')
        if (job_dir / 'cancel.flag').exists():
            raise RuntimeError('cancel.flag is present; remove it only if you intend to resume')
        legacy_no_hash = not state.get('config_sha256')
        done = verify_completed_results(state.get('results', []), plan, job_dir)
        if state.get('current', done) > done:
            raise RuntimeError('Status reports more completed runs than saved results; inspect files')
        for filename in ('status.json', 'status_recovery.json', 'job_summary.csv'):
            original = job_dir / filename
            if original.is_file():
                backup = job_dir / f'{filename}.before_resume_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.bak'
                shutil.copy2(original, backup)
        state['resume_count'] = int(state.get('resume_count', 0)) + 1
        state['pid'] = os.getpid()
        state['current'] = done
        state['total'] = total
        state['config_sha256'] = digest
        state.pop('finished_at', None)
        state.pop('error_traceback', None)
        print(f'[RESUME] {done}/{total} saved runs verified; {total - done} remain.', flush=True)
        if legacy_no_hash and done:
            print('[NOTE] Legacy job: original config hash was not recorded. '
                  'Verified saved scenario/severity/repeat order; keep original scenario.json unchanged.',
                  flush=True)
    else:
        if status_path.exists() or (job_dir / 'status_recovery.json').exists():
            prior = read_saved_state(job_dir)
            if prior.get('results') or int(prior.get('current', 0)) > 0:
                raise RuntimeError('This job already contains results. Use --resume instead '
                                   'of restarting and overwriting progress.')
            if prior.get('state') in ('running', 'preparing'):
                raise RuntimeError('Existing job appears active; refusing a second runner')
        state = {'state': 'starting', 'started_at': now(), 'updated_at': now(),
                 'pid': os.getpid(), 'current': 0, 'total': total, 'results': [],
                 'message': 'Preparing', 'config_sha256': digest, 'resume_count': 0}

    def update(**kwargs):
        state.update(kwargs)
        state['updated_at'] = now()
        atomic_json(status_path, state)

    update(state='preparing', message='Preparing source video')
    try:
        nodes = available_nodes()
        if len(nodes) < cfg['node_count']:
            raise RuntimeError(f'{cfg["node_count"]} nodes requested, but only {len(nodes)} are running: {nodes}. '
                               'Start the node containers before running the experiment.')
        for name in ('07_create_numbered_y4m.py', '08_jitsi_synchronized_quality.py',
                     '05_jitsi_video_capture.py', '09_jitsi_webrtc_stats.py',
                     '10_jitsi_joint_quality_performance.py', '11_jitsi_fault_injection_v2.py'):
            if not (ROOT / name).is_file():
                raise FileNotFoundError(f'Required existing script is missing: {name}')
        engine = perceptual_module.PerceptualEngine(cfg['perceptual_metrics'])
        source = convert_video_if_needed(cfg['video'], cfg['clip_seconds'])
        numbered, metadata = ensure_numbered(source)
        make_summary_csv(job_dir, state['results'])
        update(state='running', message='Starting Jitsi tests',
               prepared_source=str(source), numbered_source=str(numbered))
        for i in range(state['current'], total):
            cond, repeat = plan[i]
            if (job_dir / 'cancel.flag').exists():
                update(state='cancelled', message='Cancelled by user')
                return
            label = f'{cond["scenario"]}/{cond["severity"]} repeat {repeat}/{cfg["repeats"]}'
            update(message=label)
            print('\n' + '=' * 70 + '\n' + label + '\n' + '=' * 70, flush=True)
            prefix = capture_name_prefix(config.get('name') or job_dir.name, cond, repeat)
            result = run_one(source, numbered, metadata, cond, cfg['duration'], cfg['interval'], nodes,
                             capture_prefix=prefix)
            if result is None:
                raise RuntimeError('Test finished but no result was returned')
            write_capture_info(result, job_dir, config.get('name') or job_dir.name,
                               cond, repeat, i + 1)
            if engine.selected:
                update(message=f'Computing perceptual metrics: {label}')
                extra = perceptual_module.enrich_run(
                    result['run_folder'], source, metadata, engine, ROOT)
                result.update(extra)
            result['repeat_index'] = repeat
            result['plan_index'] = i + 1
            valid, reason, stalled_count = check_stream_continuity(result['run_folder'])
            result['run_valid'] = valid
            result['stalled_intervals'] = stalled_count
            result['validity_warning'] = reason
            if not valid:
                print('[WARNING] INVALID RUN: ' + reason, flush=True)
            state['results'].append(result)
            update(current=len(state['results']), message=label + ' completed')
            make_summary_csv(job_dir, state['results'])
            if state['current'] < total:
                print('Waiting 10 seconds before the next run...', flush=True)
                time.sleep(10)
        invalid = sum(not r.get('run_valid', True) for r in state['results'])
        update(state='completed', finished_at=now(),
               message=(f'All runs finished; {invalid} run(s) INVALID due to video stall'
                        if invalid else 'All experiment runs completed'))
    except Exception as exc:
        traceback.print_exc()
        update(state='failed', finished_at=now(), message=str(exc),
               error_traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--resume', action='store_true', help='Continue a failed job using saved results')
    args = parser.parse_args()
    job_dir = Path(args.job).resolve()
    if job_dir.parent != JOBS.resolve() or not job_dir.is_dir():
        parser.error('Invalid job directory')
    run(job_dir, resume=args.resume)


if __name__ == '__main__':
    main()

# VISUAL_ORACLE_WEBRTC_V04
