"""Visual Oracle v0.2/v0.3 Windows status + stalled-video hotfix.
Run from D:\\visual-oracle-revision after stopping Streamlit and active jobs.
Creates timestamped backups; refuses unknown source layouts.
"""
from pathlib import Path
from datetime import datetime
import py_compile
import shutil

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def replace_once(src, old, new, label):
    if new in src:
        print(f'[SKIP] {label}: already patched')
        return src
    if src.count(old) != 1:
        raise RuntimeError(f'Cannot safely patch {label}: expected one exact source match, found {src.count(old)}')
    return src.replace(old, new, 1)


def patch_runner(s):
    s = replace_once(s,
'''def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
''',
'''def atomic_json(path: Path, obj):
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
''', 'runner: atomic_json')
    s = replace_once(s,
'''def run(job_dir):
''',
'''def check_stream_continuity(run_folder):
    """Check receiver freshness, not just successful PNG capture/barcode decode."""
    path = Path(run_folder) / 'joint_measurements.csv'
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    longest = 0
    current = 0
    stalled = []
    for row in rows:
        try:
            fps = float(row.get('observed_decode_fps', ''))
            bitrate = float(row.get('bitrate_mbps', ''))
            is_stalled = fps <= 0.01 and bitrate <= 0.00001
        except (TypeError, ValueError):
            is_stalled = False
        if is_stalled:
            current += 1
            longest = max(longest, current)
            stalled.append(str(row.get('sample', '?')))
        else:
            current = 0
    valid = longest < 2
    reason = (f'Inbound video stalled: {longest} consecutive zero-decode/zero-bitrate '
              f'intervals (samples {", ".join(stalled)}). PNG capture alone is not proof of live video.'
              if not valid else '')
    return valid, reason, len(stalled)


def run(job_dir):
''', 'runner: stream validator')
    s = replace_once(s,
'''                result['repeat_index'] = rep + 1
                state['results'].append(result)
''',
'''                result['repeat_index'] = rep + 1
                valid, reason, stalled_count = check_stream_continuity(result['run_folder'])
                result['run_valid'] = valid
                result['stalled_intervals'] = stalled_count
                result['validity_warning'] = reason
                if not valid:
                    print('[WARNING] INVALID RUN: ' + reason, flush=True)
                state['results'].append(result)
''', 'runner: run validity flag')
    s = replace_once(s,
'''        update(state='completed', finished_at=now(), message='All experiment runs completed')
''',
'''        invalid = sum(not r.get('run_valid', True) for r in state['results'])
        update(state='completed', finished_at=now(),
               message=(f'All runs finished; {invalid} run(s) INVALID due to video stall'
                        if invalid else 'All experiment runs completed'))
''', 'runner: completion warning')
    return s


def patch_capture(s):
    s = replace_once(s,
'''            if row['capture_status'] != 'OK':
                row['quality_status'] = 'SKIPPED'
                continue
            try:
''',
'''            if row['capture_status'] != 'OK':
                row['quality_status'] = 'SKIPPED'
                continue
            # A canvas PNG can be perfectly readable while the remote video is
            # frozen. Never score a stale frame as a fresh synchronized sample.
            fps = numeric(row.get('observed_decode_fps'))
            bitrate = numeric(row.get('bitrate_mbps'))
            if fps is not None and bitrate is not None and fps <= 0.01 and bitrate <= 0.00001:
                row['quality_status'] = 'STALE'
                row['quality_error'] = 'No newly decoded video frames in preceding interval'
                continue
            try:
''', 'capture: stale frame exclusion')
    s = replace_once(s,
'''    summary = {
        **reference_info, 'samples': len(rows),
''',
'''    stale_samples = sum(r.get('quality_status') == 'STALE' for r in rows)
    summary = {
        **reference_info, 'samples': len(rows),
        'stale_samples': stale_samples,
        'warning': ('Receiver video stalled; stale captures excluded from image-quality averages'
                    if stale_samples else ''),
''', 'capture: summary stale count')
    s = replace_once(s,
r'''    print('\nSONUC - Birlesik deney', flush=True)
''',
r'''    if stale_samples:
        print(f'WARNING: {stale_samples} stale PNG sample(s) excluded; '
              'this session must be reviewed before scientific comparison.', flush=True)
    print('\nSONUC - Birlesik deney', flush=True)
''', 'capture: console warning')
    return s


def patch_dashboard(s):
    s = replace_once(s,
'''        status = read_json(path / 'status.json', {})
        scenario = read_json(path / 'scenario.json', {})
''',
'''        status_path = path / 'status.json'
        recovery_path = path / 'status_recovery.json'
        try:
            use_recovery = (recovery_path.exists() and
                            (not status_path.exists() or
                             recovery_path.stat().st_mtime_ns > status_path.stat().st_mtime_ns))
        except OSError:
            use_recovery = False
        status = read_json(recovery_path if use_recovery else status_path, {})
        scenario = read_json(path / 'scenario.json', {})
''', 'dashboard: recover status')
    s = replace_once(s,
'''    df = pd.DataFrame(results)
    if not df.empty:
''',
'''    df = pd.DataFrame(results)
    if not df.empty:
        if 'run_valid' in df.columns:
            invalid_rows = df[df['run_valid'].eq(False)]
            if not invalid_rows.empty:
                st.error(f'{len(invalid_rows)} run(s) have a stalled receiver stream. '
                         'Their image-quality values are diagnostic only; exclude from research comparisons.')
                st.dataframe(invalid_rows[[c for c in
                    ('scenario', 'severity', 'repeat_index', 'stalled_intervals', 'validity_warning')
                    if c in invalid_rows.columns]], use_container_width=True, hide_index=True)
''', 'dashboard: invalid warning')
    s = replace_once(s,
'''            comparison = df.copy()
            comparison['condition'] = comparison['scenario'].astype(str) + ' / ' + comparison['severity'].astype(str)
''',
'''            comparison = df[df['run_valid'].eq(True)].copy() if 'run_valid' in df.columns else df.copy()
            if comparison.empty:
                st.warning('No validated runs available for comparison.')
                return
            comparison['condition'] = comparison['scenario'].astype(str) + ' / ' + comparison['severity'].astype(str)
''', 'dashboard: filter invalid comparisons')
    return s


def main():
    replacements = {
        '12_visual_oracle_runner.py': patch_runner,
        '10_jitsi_joint_quality_performance.py': patch_capture,
        '12_visual_oracle_dashboard.py': patch_dashboard,
    }
    originals = {}
    patched = {}
    for name, fn in replacements.items():
        p = ROOT / name
        if not p.is_file():
            raise FileNotFoundError(f'Missing {p}; place this patch script in the project folder')
        originals[p] = p.read_text(encoding='utf-8-sig')
        patched[p] = fn(originals[p])
    # Validate all patches before writing any file.
    for p, content in patched.items():
        compile(content, str(p), 'exec')
    for p, content in patched.items():
        backup = p.with_name(p.name + f'.backup_{STAMP}')
        shutil.copy2(p, backup)
        p.write_text(content, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
        print(f'[OK] {p.name}; backup: {backup.name}')
    print('[OK] Hotfix installed. Restart Streamlit before launching new experiments.')
    print('[NOTE] Existing experimental CSV files are preserved. Previous stalled runs must be excluded manually.')


if __name__ == '__main__':
    main()
