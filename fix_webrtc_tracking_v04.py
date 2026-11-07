"""Visual Oracle validation hotfix v0.4 (compatible with dashboard v0.2/v0.3).

Fixes stale RTCPeerConnection.getStats selection, false STALE classification,
adds playback liveness diagnostics, and makes Windows status updates resilient.

Run in Anaconda Prompt from the project directory, after stopping Streamlit:
    python fix_webrtc_tracking_v04.py

This patch does not delete experiment data, source videos or scenarios.
It backs up all changed files before writing any change.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TAG = 'VISUAL_ORACLE_WEBRTC_V04'


def replace_one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected exactly one match, got {n}; no files modified')
    return s.replace(old, new, 1)


def patch_rtc(s):
    if TAG in s:
        return s
    old = '''    const q = typeof v.getVideoPlaybackQuality === 'function' ?
      v.getVideoPlaybackQuality() : null;
    return {id, parentId, readyState:v.readyState, paused:v.paused,
      width:v.videoWidth, height:v.videoHeight,
      currentTime:v.currentTime,
      totalVideoFrames:q ? q.totalVideoFrames : null,
      droppedVideoFrames:q ? q.droppedVideoFrames : null,
      webkitDecodedFrameCount:v.webkitDecodedFrameCount ?? null};'''
    new = '''    const q = typeof v.getVideoPlaybackQuality === 'function' ?
      v.getVideoPlaybackQuality() : null;
    if (!window.__visualOracleVideoUids) {
      window.__visualOracleVideoUids = new WeakMap();
      window.__visualOracleNextVideoUid = 1;
    }
    if (!window.__visualOracleVideoUids.has(v)) {
      window.__visualOracleVideoUids.set(v, window.__visualOracleNextVideoUid++);
    }
    const tracks = v.srcObject && typeof v.srcObject.getVideoTracks === 'function'
      ? v.srcObject.getVideoTracks() : [];
    return {id, parentId, domUid:window.__visualOracleVideoUids.get(v),
      srcVideoTrackIds:tracks.map(t => t.id),
      srcVideoTrackStates:tracks.map(t => t.readyState),
      readyState:v.readyState, paused:v.paused,
      width:v.videoWidth, height:v.videoHeight,
      currentTime:v.currentTime,
      totalVideoFrames:q ? q.totalVideoFrames : null,
      droppedVideoFrames:q ? q.droppedVideoFrames : null,
      webkitDecodedFrameCount:v.webkitDecodedFrameCount ?? null};'''
    s = replace_one(s, old, new, '09 video element diagnostics')
    start = s.index('def choose_inbound(')
    end = s.index('\ndef choose_remote_video(', start)
    new_function = '''def choose_inbound(data, previous_id=None, previous_records=None):
    """Select the *active* inbound stream rather than pinning a stale stats ID.

    Jitsi can replace RTCPeerConnection/inbound RTP records during a session.
    A record can remain in getStats() with constant counters while the
    replacement stream and the HTMLVideoElement continue to play.
    """
    candidates = data.get('inboundVideos', [])
    if not candidates:
        return None
    previous_records = previous_records or {}
    pcs = {p.get('index'): p for p in data.get('peerConnections', [])}
    remote_video = choose_remote_video(data)
    track_ids = set((remote_video or {}).get('srcVideoTrackIds') or [])

    def score(s):
        old = previous_records.get(s.get('id'))
        frame_growth = delta(s.get('framesDecoded'), old.get('framesDecoded')) if old else None
        byte_growth = delta(s.get('bytesReceived'), old.get('bytesReceived')) if old else None
        fps = number(s.get('framesPerSecond')) or 0
        state = pcs.get(s.get('pcIndex'), {}).get('connectionState')
        matching_track = bool(track_ids and s.get('trackIdentifier') in track_ids)
        advancing = (frame_growth is not None and frame_growth > 0) or (byte_growth is not None and byte_growth > 0)
        newly_observed_active = bool(previous_records and old is None and fps > 0 and
                                     (number(s.get('framesDecoded')) or 0) > 0)
        # Playback-track match and observed counter growth outrank historic totals.
        # Do not prefer the previously selected ID merely because it still exists.
        return (int(matching_track), int(advancing), int(newly_observed_active),
                int(state == 'connected'), int(fps > 0),
                int(s.get('id') == previous_id and advancing),
                number(frame_growth) or 0, number(byte_growth) or 0,
                number(s.get('framesDecoded')) or 0)

    return max(candidates, key=score)


'''
    s = s[:start] + new_function + s[end:]
    s += '\n# ' + TAG + '\n'
    return s


def patch_joint(s):
    if TAG in s:
        return s
    s = replace_one(s,
'''    'stats_to_capture_ms',
]''',
'''    'stats_to_capture_ms',
    'inbound_count', 'pc_index', 'video_dom_uid', 'video_track_ids',
    'playback_delta_frames', 'observed_playback_fps', 'stream_liveness',
    'stats_stream_changed', 'stats_selection_note',
]''', '10 CSV diagnostic columns')
    s = replace_one(s,
'''    selected_id, previous, previous_elapsed = None, None, None
''',
'''    selected_id, previous, previous_elapsed = None, None, None
    previous_inbounds = {}
    previous_video_count, previous_video_uid = None, None
    previous_playback_elapsed = None
''', '10 state initialization')
    s = replace_one(s,
'''                stream = rtc.choose_inbound(data, selected_id)
                video = rtc.choose_remote_video(data)
''',
'''                stream = rtc.choose_inbound(data, selected_id, previous_inbounds)
                video = rtc.choose_remote_video(data)
                row['inbound_count'] = len(data.get('inboundVideos', []))
''', '10 active stream selection')
    s = replace_one(s,
'''                if selected_id != stream.get('id'):
                    previous, previous_elapsed = None, None
                selected_id = stream['id']
''',
'''                switched = selected_id is not None and selected_id != stream.get('id')
                row['stats_stream_changed'] = bool(switched)
                if selected_id != stream.get('id'):
                    previous, previous_elapsed = None, None
                selected_id = stream['id']
                row['pc_index'] = stream.get('pcIndex')
                if switched:
                    print(f'[WebRTC] Active inbound stream switched to {selected_id}', flush=True)
''', '10 stream switch bookkeeping')
    s = replace_one(s,
'''                if video:
                    row.update(playback_width=video.get('width'),
                               playback_height=video.get('height'),
                               playback_total_frames=video.get('totalVideoFrames'))
                compute_intervals(row, stream, previous, elapsed, previous_elapsed)
                previous, previous_elapsed = stream, elapsed
''',
'''                if video:
                    row.update(playback_width=video.get('width'),
                               playback_height=video.get('height'),
                               playback_total_frames=video.get('totalVideoFrames'),
                               video_dom_uid=video.get('domUid'),
                               video_track_ids=';'.join(video.get('srcVideoTrackIds') or []))
                compute_intervals(row, stream, previous, elapsed, previous_elapsed)
                # Independent playback liveness signal. A counter reset or a
                # newly-created HTMLVideoElement is not evidence of a stall.
                current_count = rtc.number(video.get('totalVideoFrames')) if video else None
                current_uid = video.get('domUid') if video else None
                if (current_count is not None and previous_video_count is not None
                        and current_uid == previous_video_uid
                        and current_count >= previous_video_count):
                    playback_delta = current_count - previous_video_count
                    row['playback_delta_frames'] = playback_delta
                    dt_play = elapsed - previous_playback_elapsed if previous_playback_elapsed is not None else None
                    if dt_play is not None and dt_play > 0:
                        row['observed_playback_fps'] = round(playback_delta / dt_play, 4)
                else:
                    playback_delta = None
                    if previous_video_count is not None:
                        row['stats_selection_note'] = 'PLAYBACK_COUNTER_RESET_OR_VIDEO_REPLACED'
                decoded_fps = numeric(row.get('observed_decode_fps'))
                bitrate = numeric(row.get('bitrate_mbps'))
                if (playback_delta is not None and playback_delta > 0) or (decoded_fps is not None and decoded_fps > 0):
                    row['stream_liveness'] = 'LIVE'
                elif (playback_delta == 0 and decoded_fps == 0 and bitrate == 0):
                    row['stream_liveness'] = 'STALLED'
                else:
                    row['stream_liveness'] = 'UNKNOWN'
                if decoded_fps == 0 and playback_delta is not None and playback_delta > 0:
                    row['stats_selection_note'] = 'RTP_STATS_MISMATCH_PLAYBACK_ACTIVE'
                previous_inbounds = {x['id']: x for x in data.get('inboundVideos', []) if x.get('id')}
                previous, previous_elapsed = stream, elapsed
                previous_video_count, previous_video_uid = current_count, current_uid
                previous_playback_elapsed = elapsed
''', '10 independent playback tracking')
    # Old hotfix incorrectly excluded captures solely because stale RTP counters.
    stale_block = '''            # A canvas PNG can be perfectly readable while the remote video is
            # frozen. Never score a stale frame as a fresh synchronized sample.
            fps = numeric(row.get('observed_decode_fps'))
            bitrate = numeric(row.get('bitrate_mbps'))
            if fps is not None and bitrate is not None and fps <= 0.01 and bitrate <= 0.00001:
                row['quality_status'] = 'STALE'
                row['quality_error'] = 'No newly decoded video frames in preceding interval'
                continue
'''
    if stale_block in s:
        s = replace_one(s, stale_block,
'''            # Exclude only corroborated stalls; an old RTP stats record may
            # stop while the browser's video playback is still advancing.
            if row.get('stream_liveness') == 'STALLED':
                row['quality_status'] = 'STALE'
                row['quality_error'] = 'Both active RTP and HTMLVideoElement playback stopped'
                continue
''', '10 correct stale classification')
    else:
        s = replace_one(s,
'''            if row['capture_status'] != 'OK':
                row['quality_status'] = 'SKIPPED'
                continue
            try:
''',
'''            if row['capture_status'] != 'OK':
                row['quality_status'] = 'SKIPPED'
                continue
            if row.get('stream_liveness') == 'STALLED':
                row['quality_status'] = 'STALE'
                row['quality_error'] = 'Both active RTP and HTMLVideoElement playback stopped'
                continue
            try:
''', '10 correct stale classification')
    # Existing hotfix already defines stale_samples; don't redeclare.
    if '    stale_samples = sum(' not in s:
        s = replace_one(s,
'''    summary = {
        **reference_info, 'samples': len(rows),
''',
'''    stale_samples = sum(r.get('quality_status') == 'STALE' for r in rows)
    summary = {
        **reference_info, 'samples': len(rows),
        'stale_samples': stale_samples,
''', '10 summary stale samples')
    s = replace_one(s,
'''        'mean_observed_decode_fps': avg(rows, 'observed_decode_fps'),
''',
'''        'mean_observed_decode_fps': avg(rows, 'observed_decode_fps'),
        'mean_observed_playback_fps': avg(rows, 'observed_playback_fps'),
        'liveness_counts': {k: sum(r.get('stream_liveness') == k for r in rows)
                            for k in ('LIVE', 'STALLED', 'UNKNOWN')},
        'stats_stream_switches': sum(r.get('stats_stream_changed') is True for r in rows),
        'rtp_playback_mismatch_samples': sum(r.get('stats_selection_note') == 'RTP_STATS_MISMATCH_PLAYBACK_ACTIVE' for r in rows),
''', '10 summary playback diagnostics')
    s += '\n# ' + TAG + '\n'
    return s


def patch_runner(s):
    if TAG in s:
        return s
    old_atomic = '''def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
'''
    if old_atomic in s:
        s = replace_one(s, old_atomic, '''def atomic_json(path: Path, obj):
    """Retry Windows sharing violations; retain recovery state if locked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    for attempt in range(30):
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.{attempt}.tmp')
        try:
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, path)
            return
        except PermissionError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.15)
    recovery = path.with_name('status_recovery.json')
    recovery.write_text(payload, encoding='utf-8')
    print('[WARNING] Windows locked status.json; status_recovery.json written.', flush=True)
''', 'runner status update')
    elif 'def atomic_json(path: Path, obj):' not in s:
        raise RuntimeError('Runner atomic_json missing')
    # Replace earlier hotfix validator: zero RTP counters alone are not proof of stalled playback.
    start = s.find('def check_stream_continuity(run_folder):')
    if start >= 0:
        end = s.find('\ndef run(job_dir):', start)
        if end < 0:
            raise RuntimeError('Cannot locate run() after check_stream_continuity')
        s = s[:start] + '''def check_stream_continuity(run_folder):
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


''' + s[end:]
    else:
        # New validator for untouched v0.2/v0.3.
        s = replace_one(s, 'def run(job_dir):\n', '''def check_stream_continuity(run_folder):
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


def run(job_dir):
''', 'runner validator')
        s = replace_one(s,
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
''', 'runner run validity')
    s += '\n# ' + TAG + '\n'
    return s


def patch_dashboard(s):
    if TAG in s:
        return s
    if "recovery_path = path / 'status_recovery.json'" not in s:
        s = replace_one(s,
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
''', 'dashboard status recovery')
    s += '\n# ' + TAG + '\n'
    return s


def main():
    replacements = {
        '09_jitsi_webrtc_stats.py': patch_rtc,
        '10_jitsi_joint_quality_performance.py': patch_joint,
        '12_visual_oracle_runner.py': patch_runner,
        '12_visual_oracle_dashboard.py': patch_dashboard,
    }
    originals = {}
    patched = {}
    for name, fn in replacements.items():
        p = ROOT / name
        if not p.is_file():
            raise FileNotFoundError(f'Missing {p}. Place patch next to the project Python scripts.')
        originals[p] = p.read_text(encoding='utf-8-sig')
        patched[p] = fn(originals[p])
    for p, source in patched.items():
        compile(source, str(p), 'exec')
    if all(originals[p] == patched[p] for p in originals):
        print('[OK] v0.4 patch is already installed. No files changed.')
        return
    for p, source in patched.items():
        if originals[p] == source:
            continue
        backup = p.with_name(p.name + '.backup_' + STAMP)
        shutil.copy2(p, backup)
        p.write_text(source, encoding='utf-8')
        print(f'[OK] Patched {p.name} (backup: {backup.name})')
    print('[OK] Patch installed. Restart Streamlit and run one short Baseline validation.')


if __name__ == '__main__':
    main()
