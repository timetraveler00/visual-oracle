"""Visual Oracle Experiment Manager (local-only Streamlit dashboard).

Install: python -m pip install "streamlit>=1.37" pandas
Start:   streamlit run 12_visual_oracle_dashboard.py
"""
from __future__ import annotations

import base64
import csv
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import zipfile

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'captures' / 'dashboard_jobs'
SCENARIOS = ROOT / 'scenarios'
VIDEO_FOLDER = ROOT / 'videos'
CAPTURES = ROOT / 'captures'
RUNNER = ROOT / '12_visual_oracle_runner.py'

FAULT_LABELS = {
    'baseline': 'S0 · Baseline (no fault)',
    'blur': 'S1 · Gaussian blur',
    'color': 'S2 · Color distortion',
    'freeze': 'S3 · Video freeze',
    'black': 'S4 · Black screen',
    'occlusion': 'S5 · Partial occlusion',
}
SEVERITY_LABELS = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
ALL_CONDITIONS = [{'scenario': 'baseline', 'severity': 'medium'}] + [
    {'scenario': s, 'severity': v}
    for s in ('blur', 'color', 'freeze', 'black', 'occlusion')
    for v in ('low', 'medium', 'high')
]
CONDITION_OPTIONS = [
    'S0 · Baseline (no fault)' if x['scenario'] == 'baseline' else
    f'{FAULT_LABELS[x["scenario"]]} — {SEVERITY_LABELS[x["severity"]]}'
    for x in ALL_CONDITIONS
]

st.set_page_config(page_title='Visual Oracle Experiment Manager', page_icon='🎥',
                   layout='wide', initial_sidebar_state='expanded')


def init_dirs():
    for folder in (JOBS, SCENARIOS, VIDEO_FOLDER, CAPTURES):
        folder.mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return default


def find_videos():
    extensions = {'.y4m', '.mp4', '.mkv', '.mov', '.avi', '.webm'}
    found = []
    for folder in (VIDEO_FOLDER,):
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if '_numbered' in path.stem or '_fault_' in path.stem:
                continue
            found.append(path.relative_to(ROOT).as_posix())
    return sorted(set(found), key=lambda p: p.lower())


def running_nodes():
    try:
        proc = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'],
                              capture_output=True, text=True, timeout=8, encoding='utf-8')
        if proc.returncode:
            return [], proc.stderr.strip()
        names = sorted((x.strip() for x in proc.stdout.splitlines()
                        if re.fullmatch(r'selenium-chrome-\d+', x.strip())),
                       key=lambda x: int(x.rsplit('-', 1)[1]))
        return names, None
    except Exception as exc:
        return [], str(exc)


def get_jobs():
    jobs = []
    for path in JOBS.glob('job_*'):
        if not path.is_dir():
            continue
        status_path = path / 'status.json'
        recovery_path = path / 'status_recovery.json'
        try:
            use_recovery = (recovery_path.exists() and
                            (not status_path.exists() or
                             recovery_path.stat().st_mtime_ns > status_path.stat().st_mtime_ns))
        except OSError:
            use_recovery = False
        status = read_json(recovery_path if use_recovery else status_path, {})
        scenario = read_json(path / 'scenario.json', {})
        jobs.append((path, status, scenario))
    return sorted(jobs, key=lambda item: item[0].name, reverse=True)


def get_saved_scenarios():
    return sorted(SCENARIOS.glob('*.json'), key=lambda p: p.name.lower())


def slug(text):
    value = re.sub(r'[^a-zA-Z0-9_-]+', '_', text.strip())[:60].strip('_')
    return value or 'scenario'


def init_state(videos):
    defaults = {
        'scenario_name': 'Blur Severity Sweep',
        'selected_video': videos[0] if videos else '',
        'node_count': 2, 'duration': 60, 'interval': 5, 'repeats': 1,
        'clip_seconds': 10, 'perceptual_metrics': ['DISTS', 'DINOv2', 'OpenCLIP'],
        'selected_conditions': [CONDITION_OPTIONS[i] for i in (0, 1, 2, 3)],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if videos and st.session_state['selected_video'] not in videos:
        st.session_state['selected_video'] = videos[0]


def current_config():
    cond = [ALL_CONDITIONS[CONDITION_OPTIONS.index(label)]
            for label in st.session_state.selected_conditions]
    return {
        'schema_version': 1,
        'name': st.session_state.scenario_name,
        'video': st.session_state.selected_video,
        'node_count': st.session_state.node_count,
        'duration': st.session_state.duration,
        'interval': st.session_state.interval,
        'repeats': st.session_state.repeats,
        'clip_seconds': st.session_state.clip_seconds,
        'perceptual_metrics': st.session_state.perceptual_metrics,
        'conditions': cond,
        'methodology': 'Source-side content faults, two participants, sequential independent Jitsi sessions',
    }


def apply_config(config, videos):
    st.session_state.scenario_name = config.get('name', 'Scenario')
    video = config.get('video', '')
    st.session_state.selected_video = video if video in videos else (videos[0] if videos else '')
    for key, default in [('node_count', 2), ('duration', 60), ('interval', 5),
                         ('repeats', 1), ('clip_seconds', 10)]:
        st.session_state[key] = int(config.get(key, default))
    st.session_state.perceptual_metrics = [m for m in config.get('perceptual_metrics', []) if m in ('DISTS', 'DINOv2', 'DINOv2-Large', 'OpenCLIP')]
    keys = {(x['scenario'], x.get('severity', 'medium')) for x in config.get('conditions', [])}
    st.session_state.selected_conditions = [label for label, x in zip(CONDITION_OPTIONS, ALL_CONDITIONS)
                                             if (x['scenario'], x['severity']) in keys]


def launch_job(config):
    job_id = datetime.now().strftime('job_%Y%m%d_%H%M%S_') + secrets.token_hex(3)
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True)
    (job_dir / 'scenario.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (job_dir / 'status.json').write_text(json.dumps({'state': 'starting', 'current': 0, 'total': len(config['conditions']) * config['repeats'], 'results': [], 'message': 'Starting job'}, ensure_ascii=False), encoding='utf-8')
    log_handle = (job_dir / 'runner.log').open('w', encoding='utf-8', buffering=1)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    try:
        process = subprocess.Popen([sys.executable, str(RUNNER), '--job', str(job_dir)],
                                   cwd=str(ROOT), stdout=log_handle,
                                   stderr=subprocess.STDOUT, creationflags=flags)
    except Exception as exc:
        (job_dir / 'status.json').write_text(json.dumps({'state': 'failed', 'message': str(exc), 'results': [], 'current': 0, 'total': 0}, ensure_ascii=False), encoding='utf-8')
        raise
    finally:
        log_handle.close()
    (job_dir / 'launcher.json').write_text(json.dumps({'pid': process.pid,
        'started_at': datetime.now().isoformat(), 'command': '12_visual_oracle_runner.py'}), encoding='utf-8')
    return job_dir


def is_busy():
    return any(status.get('state') in ('starting', 'preparing', 'running')
               for _, status, _ in get_jobs())


def render_job_status():
    jobs = get_jobs()
    if not jobs:
        st.info('No experiments have been run yet.')
        return
    job_dir, status, config = jobs[0]
    state = status.get('state', 'starting')
    labels = {'starting': 'Starting', 'preparing': 'Preparing video',
              'running': 'Running', 'completed': 'Completed',
              'failed': 'Failed', 'cancelled': 'Cancelled'}
    st.markdown(f'**Latest job:** `{job_dir.name}` — **{labels.get(state, state)}**')
    total, current = status.get('total', 0), status.get('current', 0)
    st.progress(current / total if total else 0, text=f'{current}/{total} runs · {status.get("message", "")}'[:140])
    if state == 'failed':
        st.error(status.get('message', 'Unknown error'))
    elif state == 'completed':
        st.success('All scenario runs completed.')
    if state in ('starting', 'preparing', 'running'):
        if st.button('Stop after the current run', key='stop_job'):
            (job_dir / 'cancel.flag').write_text('stop', encoding='utf-8')
            st.warning('Cancellation requested; the current run will finish first.')
    log_path = job_dir / 'runner.log'
    if log_path.exists():
        with st.expander('Live execution log', expanded=state in ('running', 'failed')):
            text = log_path.read_text(encoding='utf-8', errors='replace')
            st.code('\n'.join(text.splitlines()[-70:]) or 'No output yet', language='text')


def load_measurements(run_dir):
    for name in ('enhanced_measurements.csv', 'fault_measurements.csv', 'joint_measurements.csv'):
        path = run_dir / name
        if path.is_file():
            return pd.read_csv(path, encoding='utf-8-sig'), path
    return None, None


def numeric_column(df, key):
    return pd.to_numeric(df[key], errors='coerce') if key in df else pd.Series(dtype=float)


def render_graph(df, x, keys, label):
    available = [key for key in keys if key in df.columns]
    if not available or x not in df.columns:
        return
    view = df[[x] + available].copy()
    for col in [x] + available:
        view[col] = pd.to_numeric(view[col], errors='coerce')
    view = view.dropna(subset=[x])
    if not view.empty:
        st.markdown(f'**{label}**')
        st.line_chart(view.set_index(x)[available], use_container_width=True)


def locate_reference(settings):
    name = settings.get('reference_video', '')
    if not name or Path(name).name != name:
        return None
    for folder in (ROOT, VIDEO_FOLDER, ROOT / 'prepared_videos'):
        path = folder / name
        if path.is_file():
            return path
    return None


def preview_run(run_dir, df):
    pngs = [p for p in sorted(run_dir.glob('Participant_B_remote_*.png')) if p.is_file()]
    if not pngs:
        st.info('No captured video frames are available for this run.')
        return
    files = {p.name: p for p in pngs}
    selected = st.selectbox('Captured frame', list(files), key='preview_' + run_dir.name)
    row = None
    if 'capture_filename' in df.columns:
        subset = df[df['capture_filename'] == selected]
        if not subset.empty:
            row = subset.iloc[0]
    left, right = st.columns(2)
    with right:
        st.image(str(files[selected]), caption='Participant_B — actual received frame', use_container_width=True)
    with left:
        reference = locate_reference(read_json(run_dir / 'settings.json', {}))
        if reference is not None and row is not None and pd.notna(row.get('frame_index')):
            try:
                import cv2
                import importlib.util
                spec = importlib.util.spec_from_file_location('quality_preview08', ROOT / '08_jitsi_synchronized_quality.py')
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                w, h, fps, offsets = mod.index_y4m(reference)
                frame_index = int(row['frame_index'])
                rec = cv2.imread(str(files[selected]))
                with reference.open('rb') as f:
                    bgr = mod.read_reference(f, offsets[frame_index], w, h, rec.shape[1], rec.shape[0])
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                         caption=f'Clean reference — FRAME {frame_index}', use_container_width=True)
            except Exception as exc:
                st.info('Could not render reference preview: ' + str(exc))
        else:
            st.info('Reference video is unavailable for this run.')
    if row is not None:
        parts = []
        for key, label in [('ssim_gray_masked', 'SSIM'), ('psnr_rgb_masked_db', 'PSNR (dB)'),
                           ('dists_similarity', 'DISTS'), ('dinov2_cosine', 'DINOv2'),
                           ('openclip_cosine', 'OpenCLIP'),
                          ('dinov2_large_cosine', 'DINOv2-Large'),
                           ('observed_decode_fps', 'FPS'), ('bitrate_mbps', 'Mbps'),
                           ('jitter_ms', 'Jitter (ms)')]:
            if key in row and pd.notna(row[key]):
                try:
                    parts.append(f'{label}: {float(row[key]):.3f}')
                except (ValueError, TypeError):
                    pass
        st.caption(' · '.join(parts))


def html_report(job_dir, status, config):
    rows = status.get('results', [])
    header = ['Condition', 'Repeat', 'SSIM (all)', 'SSIM (fault active)', 'PSNR (active)', 'ΔE76 (active)', 'FPS', 'Mbps', 'DISTS', 'DINOv2', 'DINOv2-Large', 'OpenCLIP']
    cells = []
    for r in rows:
        values = [r.get('scenario'), r.get('repeat_index'), r.get('mean_ssim_all'),
                  r.get('mean_ssim_fault_active'), r.get('mean_psnr_fault_active_db'),
                  r.get('mean_delta_e76_fault_active'), r.get('mean_decode_fps'),
                  r.get('mean_bitrate_mbps'), r.get('mean_dists_similarity_all'),
                  r.get('mean_dinov2_cosine_all'), r.get('mean_dinov2_large_cosine_all'), r.get('mean_openclip_cosine_all')]
        cells.append('<tr>' + ''.join(f'<td>{html.escape(str(v if v is not None else "—"))}</td>' for v in values) + '</tr>')
    table = '<table><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in header) + '</tr></thead><tbody>' + ''.join(cells) + '</tbody></table>'
    sections = []
    for r in rows:
        folder = Path(r.get('run_folder', ''))
        if not folder.is_dir() or folder.parent.resolve() != CAPTURES.resolve():
            continue
        df, _ = load_measurements(folder)
        if df is None or 'elapsed_s' not in df:
            continue
        coords = []
        if 'ssim_gray_masked' in df:
            xs = numeric_column(df, 'elapsed_s')
            ys = numeric_column(df, 'ssim_gray_masked')
            for x, y in zip(xs, ys):
                if pd.notna(x) and pd.notna(y):
                    coords.append((float(x), float(y)))
        if len(coords) > 1:
            max_x = max(x for x, _ in coords) or 1
            pts = ' '.join(f'{35 + x/max_x*620:.1f},{230 - min(1,max(0,y))*190:.1f}' for x,y in coords)
            svg = f'<svg viewBox="0 0 690 260" role="img" aria-label="SSIM over time"><line x1="35" y1="40" x2="35" y2="230" stroke="#aaa"/><line x1="35" y1="230" x2="655" y2="230" stroke="#aaa"/><polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="3"/><text x="40" y="26">SSIM / time</text><text x="40" y="247">0 s</text><text x="610" y="247">{max_x:g} s</text></svg>'
        else:
            svg = ''
        pngs = sorted(folder.glob('Participant_B_remote_*.png'))
        thumb = ''
        if pngs:
            try:
                import cv2
                img = cv2.imread(str(pngs[0]))
                if img is not None:
                    width = 560
                    img = cv2.resize(img, (width, round(img.shape[0]*width/img.shape[1])))
                    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if ok:
                        thumb = '<img alt="Received frame preview" style="max-width:560px;width:100%" src="data:image/jpeg;base64,' + base64.b64encode(enc.tobytes()).decode() + '">'
            except Exception:
                pass
        sections.append(f'<section><h2>{html.escape(str(r.get("scenario")))} / {html.escape(str(r.get("severity")))}, repeat {r.get("repeat_index")}</h2>{svg}{thumb}</section>')
    style = '''body{font-family:Arial,sans-serif;margin:40px auto;max-width:1000px;color:#17202a;padding:0 20px}h1{color:#123c65}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d8e0e8;padding:9px;text-align:left}th{background:#eef4fa}section{margin:30px 0;padding-top:20px;border-top:1px solid #ddd}small{color:#555}'''
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Visual Oracle Experiment Report</title><style>{style}</style><h1>Visual Oracle Experiment Report</h1><p><b>Scenario:</b> {html.escape(str(config.get("name", "")))}<br><b>Video:</b> {html.escape(str(config.get("video", "")))}<br><b>Status:</b> {html.escape(str(status.get("state", "")))}</p>{table}{"".join(sections)}<p><small>Source-side visual faults are injected before transmission. Frames within one session are correlated and are not independent experimental replicates. WebRTC FPS/bitrate measurements summarize the preceding interval.</small></p></html>'


def render_reports():
    jobs = get_jobs()
    completed = [(p, s, c) for p, s, c in jobs if s.get('results')]
    if not completed:
        st.info('Run at least one experiment to generate reports.')
        return
    labels = {p.name: (p, s, c) for p, s, c in completed}
    selected = st.selectbox('Experiment job', list(labels), key='report_job')
    job_dir, status, config = labels[selected]
    results = status.get('results', [])
    df = pd.DataFrame(results)
    if not df.empty:
        if 'run_valid' in df.columns:
            invalid_rows = df[df['run_valid'].eq(False)]
            if not invalid_rows.empty:
                st.error(f'{len(invalid_rows)} run(s) have a stalled receiver stream. '
                         'Their image-quality values are diagnostic only; exclude from research comparisons.')
                st.dataframe(invalid_rows[[c for c in
                    ('scenario', 'severity', 'repeat_index', 'stalled_intervals', 'validity_warning')
                    if c in invalid_rows.columns]], use_container_width=True, hide_index=True)
        fields = [x for x in ['scenario', 'severity', 'repeat_index', 'quality_ok', 'sample_count',
                              'mean_ssim_all', 'mean_ssim_fault_active',
                              'mean_psnr_fault_active_db', 'mean_delta_e76_fault_active',
                              'mean_decode_fps', 'mean_bitrate_mbps',
                              'mean_dists_similarity_all', 'mean_dinov2_cosine_all',
                              'mean_dinov2_large_cosine_all', 'mean_openclip_cosine_all'] if x in df.columns]
        st.dataframe(df[fields], use_container_width=True, hide_index=True)
        # Compare condition-level means; do not treat individual frames as independent trials.
        metric_keys = [k for k in ('mean_ssim_all', 'mean_dists_similarity_all',
                                   'mean_dinov2_cosine_all', 'mean_dinov2_large_cosine_all', 'mean_openclip_cosine_all')
                       if k in df.columns and pd.to_numeric(df[k], errors='coerce').notna().any()]
        if metric_keys and 'scenario' in df and 'severity' in df:
            comparison = df[df['run_valid'].eq(True)].copy() if 'run_valid' in df.columns else df.copy()
            if comparison.empty:
                st.warning('No validated runs available for comparison.')
                return
            comparison['condition'] = comparison['scenario'].astype(str) + ' / ' + comparison['severity'].astype(str)
            for k in metric_keys:
                comparison[k] = pd.to_numeric(comparison[k], errors='coerce')
            comparison = comparison.groupby('condition', sort=False)[metric_keys].mean()
            st.markdown('**Mean similarity by condition (independent sessions)**')
            st.bar_chart(comparison, use_container_width=True)
        a, b = st.columns(2)
        with a:
            st.download_button('Download job summary CSV', df[fields].to_csv(index=False).encode('utf-8-sig'),
                               file_name='visual_oracle_job_summary.csv', mime='text/csv')
        with b:
            st.download_button('Download HTML research report', html_report(job_dir, status, config),
                               file_name=f'{job_dir.name}_report.html', mime='text/html')
    run_map = {f'{i+1}. {r.get("scenario")}/{r.get("severity")} · repeat {r.get("repeat_index", 1)}': Path(r['run_folder'])
               for i, r in enumerate(results) if 'run_folder' in r}
    if not run_map:
        return
    label = st.selectbox('Detailed run', list(run_map), key='selected_run')
    run_dir = run_map[label]
    if not run_dir.is_dir() or run_dir.parent.resolve() != CAPTURES.resolve():
        st.warning('Run folder not found.')
        return
    measurements, csv_path = load_measurements(run_dir)
    if measurements is None:
        st.warning('Measurement CSV file not found.')
        return
    metrics = st.columns(4)
    for col, (key, name, fmt) in zip(metrics, [
        ('ssim_gray_masked', 'SSIM', '.4f'), ('psnr_rgb_masked_db', 'PSNR (dB)', '.2f'),
        ('observed_decode_fps', 'Decode FPS', '.2f'), ('bitrate_mbps', 'Bitrate (Mbps)', '.2f')]):
        val = numeric_column(measurements, key).mean()
        col.metric(name, format(float(val), fmt) if pd.notna(val) else '—')
    perceptual_columns = [('dists_similarity', 'DISTS similarity'),
                          ('dinov2_cosine', 'DINOv2 cosine'),
                          ('openclip_cosine', 'OpenCLIP cosine'),
                          ('dinov2_large_cosine', 'DINOv2-Large cosine')]
    available_perceptual = [(key, label) for key, label in perceptual_columns
                            if key in measurements.columns and numeric_column(measurements, key).notna().any()]
    if available_perceptual:
        cols = st.columns(len(available_perceptual))
        for col, (key, label) in zip(cols, available_perceptual):
            col.metric(label, f'{numeric_column(measurements, key).mean():.4f}')
    g1, g2 = st.columns(2)
    with g1:
        render_graph(measurements, 'elapsed_s', ['ssim_gray_masked'], 'SSIM')
        render_graph(measurements, 'elapsed_s', ['observed_decode_fps'], 'Decode FPS')
        for key, label in available_perceptual:
            render_graph(measurements, 'elapsed_s', [key], label)
    with g2:
        render_graph(measurements, 'elapsed_s', ['psnr_rgb_masked_db'], 'PSNR (dB)')
        render_graph(measurements, 'elapsed_s', ['bitrate_mbps'], 'Bitrate (Mbps)')
        render_graph(measurements, 'elapsed_s', ['jitter_ms'], 'Jitter (ms)')
    st.subheader('Frame previews')
    preview_run(run_dir, measurements)
    st.download_button('Download measurements CSV', csv_path.read_bytes(),
                       file_name=f'{run_dir.name}_{csv_path.name}', mime='text/csv')
    with st.expander('Raw measurements'):
        st.dataframe(measurements, use_container_width=True, hide_index=True)
    metric_details = read_json(run_dir / 'perceptual_summary.json', None)
    if metric_details:
        with st.expander('Perceptual metric methodology'):
            st.json(metric_details)



def render_source_preview(relative):
    if not relative:
        return
    source = ROOT / relative
    if not source.is_file():
        return
    with st.expander('Selected video — preview and file details'):
        st.caption(f'{relative} · {source.stat().st_size / 1024**2:,.1f} MiB')
        if source.suffix.lower() != '.y4m':
            st.video(str(source))
            st.caption('This file will be converted to 720p/30 FPS Y4M with FFmpeg before running.')
            return
        try:
            import cv2
            import numpy as np
            with source.open('rb') as f:
                header = f.readline().decode('ascii')
                w = int(re.search(r'\bW(\d+)', header).group(1))
                h = int(re.search(r'\bH(\d+)', header).group(1))
                fps = re.search(r'\bF(\d+):(\d+)', header)
                f.readline()  # first FRAME marker
                payload = f.read(w*h*3//2)
            if len(payload) != w*h*3//2:
                raise ValueError('First frame is incomplete')
            yuv = np.frombuffer(payload, np.uint8).reshape(h*3//2, w)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f'{w}×{h} · {int(fps.group(1))/int(fps.group(2)):.2f} FPS · Y4M')
        except Exception as exc:
            st.warning('Video preview unavailable: ' + str(exc))

def render_history():
    all_runs = CAPTURES / 'fault_runs_summary.csv'
    if not all_runs.is_file():
        st.info('No fault experiment summary is available yet.')
        return
    try:
        df = pd.read_csv(all_runs, encoding='utf-8-sig')
    except Exception as exc:
        st.error(str(exc))
        return
    if df.empty:
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    if 'scenario' in df and 'severity' in df:
        columns = [c for c in ('mean_ssim_all', 'mean_decode_fps', 'mean_bitrate_mbps') if c in df]
        if columns:
            for col in columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            grouped = df.groupby(['scenario', 'severity'], dropna=False)[columns].agg(['mean', 'std', 'count'])
            st.markdown('**Independent sessions: mean, standard deviation and repeat count**')
            st.dataframe(grouped, use_container_width=True)
    st.caption('Frames within one session are correlated; distinct sessions count as independent repetitions.')


init_dirs()
videos = find_videos()
init_state(videos)
st.title('Visual Oracle Experiment Manager')
st.caption('Jitsi Meet · Selenium Grid · Synchronized SSIM/PSNR · WebRTC statistics · Controlled source-video faults · Perceptual metrics')

with st.sidebar:
    st.header('Experiment configuration')
    saved = get_saved_scenarios()
    saved_options = ['— Select —'] + [p.name for p in saved]
    selected_saved = st.selectbox('Load saved scenario', saved_options)
    if st.button('Load scenario', disabled=selected_saved == '— Select —'):
        config = read_json(SCENARIOS / selected_saved, {})
        apply_config(config, videos)
        st.rerun()
    st.text_input('Scenario name', key='scenario_name')
    if videos:
        st.selectbox('Source video', videos, key='selected_video')
    else:
        st.error('Place Y4M / MP4 / MKV / MOV / AVI / WEBM source files in the videos folder.')
    st.number_input('Selenium node capacity (2–10)', min_value=2, max_value=10,
                    step=1, key='node_count')
    st.number_input('Measurement duration (seconds)', min_value=5, max_value=3600,
                    step=5, key='duration')
    st.selectbox('Sampling interval (seconds)', [1, 2, 5, 10, 15, 20, 30, 60],
                 key='interval')
    st.number_input('Independent repetitions per condition', min_value=1, max_value=20,
                    step=1, key='repeats')
    st.number_input('MP4 → Y4M clip length (seconds)', min_value=3, max_value=60,
                    step=1, key='clip_seconds')
    st.multiselect('Experiment conditions', CONDITION_OPTIONS,
                   key='selected_conditions')
    st.multiselect('Perceptual similarity metrics (optional)',
                   ['DISTS', 'DINOv2', 'DINOv2-Large', 'OpenCLIP'], key='perceptual_metrics',
                   help='Pretrained models are downloaded on first use; CPU processing can take several minutes. DISTS=1−distance; DINOv2 and OpenCLIP=cosine similarity.')
    st.caption('SSIM, PSNR, DeltaE76 and WebRTC statistics are always collected. Optional perceptual metrics are computed offline after capture.')
    st.caption('Conditions execute sequentially. Each run uses two Selenium sessions: Participant_A and Participant_B.')
    config = current_config()
    total = len(config['conditions']) * config['repeats']
    st.caption(f'{total} independent sessions · approx. {total * (config["duration"] + 30) // 60} minutes plus setup/connection time')
    if st.button('Save scenario', use_container_width=True):
        if not config['conditions'] or not config['video']:
            st.error('Select a source video and at least one condition.')
        else:
            path = SCENARIOS / (slug(config['name']) + '.json')
            if path.exists():
                path = SCENARIOS / (slug(config['name']) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
            st.success(f'Saved: {path.name}')
    if st.button('▶ Run scenario', type='primary', use_container_width=True,
                 disabled=not videos or not config['conditions'] or is_busy()):
        if config['duration'] % config['interval']:
            st.error('Measurement duration must be divisible by the sampling interval.')
        else:
            nodes, err = running_nodes()
            if err:
                st.error('Docker access failed: ' + err)
            elif len(nodes) < config['node_count']:
                st.error(f'{config["node_count"]} nodes selected; only {len(nodes)} nodes are running. Start the required Docker containers first.')
            else:
                try:
                    job = launch_job(config)
                    st.success(f'Job started: {job.name}')
                    st.rerun()
                except Exception as exc:
                    st.error('Could not start job: ' + str(exc))

nodes, docker_error = running_nodes()
c1, c2, c3 = st.columns(3)
c1.metric('Running Selenium nodes', str(len(nodes)))
c2.metric('Selected node capacity', str(config['node_count']))
c3.metric('Planned runs', str(total))
if docker_error:
    st.warning('Could not read Docker status: ' + docker_error)
elif len(nodes) < config['node_count']:
    st.warning(f'Insufficient node capacity. Running: {", ".join(nodes) or "none"}. Start additional Selenium node containers in Docker.')
else:
    st.caption('Grid node containers: ' + ', '.join(nodes))

render_source_preview(config['video'])

main_tabs = st.tabs(['Run monitor', 'Reports & charts', 'Experiment history', 'Setup & methodology'])
with main_tabs[0]:
    if hasattr(st, 'fragment'):
        @st.fragment(run_every='3s')
        def live_status():
            render_job_status()
        live_status()
    else:
        render_job_status()
        if st.button('Refresh status'):
            st.rerun()
with main_tabs[1]:
    render_reports()
with main_tabs[2]:
    render_history()
with main_tabs[3]:
    st.markdown("""
**Start:** `python -m streamlit run 12_visual_oracle_dashboard.py` (Windows Anaconda Prompt).

**Source video directory:** `D:\visual-oracle-revision\videos`. Only original videos directly in this folder are selectable.
Supported formats: Y4M, MP4, MKV, MOV, AVI, WEBM. FFmpeg is required for non-Y4M formats.
Converted videos are normalized to 1280×720, 30 FPS, YUV420; original Y4M files are preserved.

**Selenium nodes:** Configure 2–10 pre-existing Docker node containers. Each run currently uses **two participants**;
additional nodes are not used for parallel testing. All experiment conditions and repeats run sequentially.

**Fault injection:** Source-side image content manipulation, not network packet manipulation.
Frame barcode overlays are added after injection and excluded from quality evaluation.
Freeze and black-screen windows may be missed by sparse sampling; choose a shorter sampling interval when needed.

**Metrics:** SSIM, PSNR, DeltaE76 and WebRTC statistics are always available.
Optional DISTS, DINOv2 ViT-B/14 (optional ViT-L/14), and OpenCLIP ViT-B-32 (LAION-2B) are computed **after capture**.
DISTS similarity is `1 − DISTS distance`; DINOv2 and OpenCLIP values are **raw cosine similarities**, not probabilities.
The perceptual models compare identical overlay-free lower crops; SSIM and PSNR exclude the overlay using a mask.
Their numerical scales and image regions are not identical and should not be treated as directly interchangeable.
First-time model setup downloads pretrained weights. CPU-only inference is supported but can be slow.

**Scientific interpretation:** Captured frames from one session are correlated observations, not independent
experimental repetitions. Receiver resolution must be controlled or stratified when comparing conditions.
Saved scenario JSON files are in `scenarios`; experiment jobs are in `captures/dashboard_jobs`.
Raw receiver captures and measurements remain in `captures/joint_*`.
""")

# VISUAL_ORACLE_WEBRTC_V04
