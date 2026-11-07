"""09 — Jitsi / Selenium Grid: measure receiver-side WebRTC video statistics.

Run in Windows Anaconda Prompt:
    cd /d D:\\visual-oracle-revision
    python 09_jitsi_webrtc_stats.py

Requires: selenium; running Selenium Grid + Jitsi; numbered Y4M present in BOTH
Chrome nodes at /tmp/720p50_shields_ter_numbered.y4m.

This script instruments RTCPeerConnection BEFORE Jitsi loads via Chrome DevTools.
It measures Participant_B's actual inbound-video RTP stats, not source FPS.
No SSIM is calculated; no camera content is uploaded.
"""

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import StaleElementReferenceException

GRID_URL = "http://localhost:4444"
JITSI_URL = "https://host.docker.internal:8443/VisualOracleLab"
Y4M_PATH_IN_CONTAINER = "/tmp/720p50_shields_ter_numbered.y4m"
DURATION_SECONDS = 60
SAMPLE_INTERVAL_SECONDS = 5
WARMUP_SECONDS = 5
OUTPUT_ROOT = Path("captures")

NAME_SELECTORS = (
    "#premeeting-name-input",
    "input[placeholder*='name' i]",
    "input[type='text']",
    "input:not([type])",
)
JOIN_XPATH = (
    "//button[contains(translate(normalize-space(.), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'join meeting')]"
    " | //*[@role='button'][contains(translate(normalize-space(.), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'join meeting')]"
)

# The CDP preload hook is injected before navigation. Proxy preserves the
# original RTCPeerConnection constructor, prototype and static methods.
PC_HOOK_JS = r"""
(() => {
  if (window.__visualOraclePcHookInstalled) return;
  window.__visualOraclePcHookInstalled = true;
  window.__visualOraclePCs = [];
  const Native = window.RTCPeerConnection || window.webkitRTCPeerConnection;
  if (!Native) { window.__visualOracleHookError = 'RTCPeerConnection missing'; return; }
  const Wrapped = new Proxy(Native, {
    construct(target, args, newTarget) {
      const pc = Reflect.construct(target, args, newTarget);
      window.__visualOraclePCs.push(pc);
      return pc;
    }
  });
  window.RTCPeerConnection = Wrapped;
  if (window.webkitRTCPeerConnection === Native) {
    window.webkitRTCPeerConnection = Wrapped;
  }
})();
"""

# One execute_async_script call obtains all active peer connections and their
# inbound-rtp/video records. The remote HTML video element is a secondary,
# independently observed playback measurement.
STATS_JS = r"""
const done = arguments[arguments.length - 1];
(async () => {
  const pcs = window.__visualOraclePCs || [];
  const records = [];
  const peerConnections = [];
  for (let i = 0; i < pcs.length; i++) {
    const pc = pcs[i];
    peerConnections.push({index:i, connectionState:pc.connectionState,
      iceConnectionState:pc.iceConnectionState, signalingState:pc.signalingState});
    if (pc.connectionState === 'closed') continue;
    try {
      const stats = await pc.getStats();
      stats.forEach(s => {
        if (s.type !== 'inbound-rtp' || (s.kind !== 'video' && s.mediaType !== 'video')) return;
        const codec = s.codecId ? stats.get(s.codecId) : null;
        const transport = s.transportId ? stats.get(s.transportId) : null;
        records.push({
          pcIndex:i, id:s.id, timestamp:s.timestamp,
          kind:s.kind || s.mediaType || '', ssrc:s.ssrc ?? null,
          trackIdentifier:s.trackIdentifier ?? null, mid:s.mid ?? null,
          codecMimeType:codec ? codec.mimeType : null,
          codecPayloadType:codec ? codec.payloadType : null,
          transportId:s.transportId ?? null,
          bytesReceived:s.bytesReceived ?? null,
          packetsReceived:s.packetsReceived ?? null,
          packetsLost:s.packetsLost ?? null,
          jitter:s.jitter ?? null,
          jitterBufferDelay:s.jitterBufferDelay ?? null,
          jitterBufferEmittedCount:s.jitterBufferEmittedCount ?? null,
          framesReceived:s.framesReceived ?? null,
          framesDecoded:s.framesDecoded ?? null,
          framesDropped:s.framesDropped ?? null,
          framesPerSecond:s.framesPerSecond ?? null,
          frameWidth:s.frameWidth ?? null, frameHeight:s.frameHeight ?? null,
          totalDecodeTime:s.totalDecodeTime ?? null,
          freezeCount:s.freezeCount ?? null,
          totalFreezesDuration:s.totalFreezesDuration ?? null,
          pliCount:s.pliCount ?? null, nackCount:s.nackCount ?? null,
          firCount:s.firCount ?? null,
          keyFramesDecoded:s.keyFramesDecoded ?? null,
          retransmittedPacketsReceived:s.retransmittedPacketsReceived ?? null,
          estimatedPlayoutTimestamp:s.estimatedPlayoutTimestamp ?? null,
          transportBytesReceived:transport ? transport.bytesReceived : null
        });
      });
    } catch (e) { peerConnections[peerConnections.length - 1].statsError = String(e); }
  }
  const videos = [...document.querySelectorAll('video')];
  const remoteVideos = videos.map(v => {
    const parent = v.closest('.videocontainer, .remote-video-container');
    const id = v.id || '', cls = String(v.className || '');
    const parentId = parent ? parent.id || '' : '';
    const parentCls = parent ? String(parent.className || '') : '';
    const remote = /remote/i.test(id) || /remote/i.test(cls) ||
      /remote/i.test(parentId) || /remote/i.test(parentCls) ||
      /^participant_/.test(parentId);
    const local = /local/i.test(id) || /local/i.test(cls) ||
      /local/i.test(parentId) || /local/i.test(parentCls);
    if (!remote || local) return null;
    const q = typeof v.getVideoPlaybackQuality === 'function' ?
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
      webkitDecodedFrameCount:v.webkitDecodedFrameCount ?? null};
  }).filter(Boolean);
  done({ok:true, capturedAtBrowser:new Date().toISOString(),
    hookInstalled:!!window.__visualOraclePcHookInstalled,
    hookError:window.__visualOracleHookError || null,
    pcCount:pcs.length, peerConnections, inboundVideos:records, remoteVideos});
})().catch(e => done({ok:false, error:String(e)}));
"""


def find_name_input(driver):
    """Find a fresh, usable name field after Jitsi DOM re-renders."""
    for selector in NAME_SELECTORS:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            try:
                if element.is_displayed() and element.is_enabled():
                    return element
            except StaleElementReferenceException:
                continue
    return None


def find_join_button(driver):
    """Never return a stale pre-meeting button reference."""
    for element in driver.find_elements(By.XPATH, JOIN_XPATH):
        try:
            if element.is_displayed() and element.is_enabled():
                return element
        except StaleElementReferenceException:
            continue
    return None


def join_meeting(driver, name):
    """Enter the participant name, CLICK Join, THEN await button disappearance.

    The receiver-side decoded-video check remains the final proof of a working
    conference. This helper only verifies the pre-meeting UI transition.
    """
    from selenium.common.exceptions import TimeoutException

    def enter_name(d):
        field = find_name_input(d)
        if field is None:
            return False
        field.clear()
        field.send_keys(name)
        return True

    try:
        WebDriverWait(driver, 40, ignored_exceptions=(StaleElementReferenceException,)).until(enter_name)

        def click_join(d):
            button = find_join_button(d)
            if button is None:
                return False
            button.click()
            return True

        # Important: do not wait for Join to disappear before clicking it.
        WebDriverWait(driver, 20, ignored_exceptions=(StaleElementReferenceException,)).until(click_join)
        WebDriverWait(driver, 50, ignored_exceptions=(StaleElementReferenceException,)).until(
            lambda d: find_join_button(d) is None
        )
    except Exception as exc:
        # Preserve original exception; diagnostics must not mask the failure.
        try:
            import re
            debug_dir = Path(__file__).resolve().parent / 'captures' / 'join_debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
            prefix = debug_dir / f'{stamp}_{safe_name}'
            driver.save_screenshot(str(prefix) + '.png')
            (Path(str(prefix) + '.html')).write_text(driver.page_source, encoding='utf-8')
            print(f'[JOIN ERROR] {type(exc).__name__}: {exc}; screenshot={prefix}.png', flush=True)
        except Exception as diagnostic_exc:
            print(f'[JOIN ERROR] Diagnostic capture failed: {diagnostic_exc}', flush=True)
        raise
    print(f'{name}: joined the meeting (UI).', flush=True)


def start_participant(name):
    options = Options()
    for flag in (
        "--ignore-certificate-errors",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,720",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
    ):
        options.add_argument(flag)
    if name == "Participant_A":
        options.add_argument(f"--use-file-for-fake-video-capture={Y4M_PATH_IN_CONTAINER}")
    options.page_load_strategy = "eager"
    driver = webdriver.Remote(command_executor=GRID_URL, options=options)
    driver.set_script_timeout(30)
    try:
        # This must run BEFORE driver.get(). Otherwise Jitsi can create PCs
        # before the hook exists and no real RTP measurements are possible.
        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument", {"source": PC_HOOK_JS}
            )
        except Exception as exc:
            raise RuntimeError(
                "Chrome DevTools on Selenium Grid is unavailable; "
                "cannot instrument RTCPeerConnection before page load. "
                f"Details: {exc}"
            ) from exc
        driver.get(JITSI_URL)
        join_meeting(driver, name)
        return driver
    except Exception:
        driver.quit()
        raise


def fetch_stats(driver):
    data = driver.execute_async_script(STATS_JS)
    if not data or not data.get("ok"):
        raise RuntimeError(f"getStats failed: {data}")
    return data


def number(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def delta(new, old):
    a, b = number(new), number(old)
    if a is None or b is None or a < b:
        return None
    return a - b


def divide(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def fmt(value, decimals=2):
    value = number(value)
    return "N/A" if value is None else f"{value:.{decimals}f}"


def choose_inbound(data, previous_id=None, previous_records=None):
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



def choose_remote_video(data):
    candidates = data.get("remoteVideos", [])
    if not candidates:
        return None
    return max(candidates, key=lambda v: (
        v.get("readyState", 0) >= 2,
        (v.get("width") or 0) * (v.get("height") or 0),
    ))


COLUMNS = [
    "sample", "elapsed_s", "timestamp_local", "status", "pc_count", "inbound_count",
    "pc_index", "inbound_id", "ssrc", "codec", "connection_state",
    "frame_width", "frame_height", "reported_fps",
    "frames_received", "frames_decoded", "frames_dropped", "packets_received",
    "packets_lost", "bytes_received", "jitter_ms", "jitter_buffer_delay_total_s",
    "jitter_buffer_emitted_count", "freeze_count", "total_freezes_duration_s",
    "remote_video_width", "remote_video_height", "playback_total_frames", "playback_dropped_frames",
    "observed_decoded_fps", "observed_received_fps", "observed_playback_fps",
    "bitrate_mbps", "interval_packet_loss_pct", "interval_jitter_buffer_ms_per_frame",
    "interval_decode_ms_per_frame", "interval_dropped_frames", "interval_freeze_count",
    "error",
]


def main():
    run_dir = OUTPUT_ROOT / datetime.now().strftime("webrtc_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "settings.json").write_text(json.dumps({
        "grid": GRID_URL, "jitsi": JITSI_URL,
        "source_video": Y4M_PATH_IN_CONTAINER,
        "duration_seconds": DURATION_SECONDS,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "measurement_side": "Participant_B receiver",
        "measurement_method": "inbound-rtp via RTCPeerConnection.getStats()",
        "note": "Browser/WebRTC reports actual receiver stats, not source file properties",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Output directory:", run_dir.resolve(), flush=True)
    drivers = {}
    rows = []
    selected_id = None
    previous = None
    previous_video = None
    previous_elapsed = None
    try:
        for name in ("Participant_A", "Participant_B"):
            print(name, "starting...", flush=True)
            drivers[name] = start_participant(name)
        receiver = drivers["Participant_B"]
        print("Waiting for receiver WebRTC stream (up to 75 seconds)...", flush=True)

        def is_ready(d):
            result = fetch_stats(d)
            inbound = choose_inbound(result)
            video = choose_remote_video(result)
            return bool(inbound and number(inbound.get("framesDecoded")) and
                        video and video.get("readyState", 0) >= 2 and
                        video.get("width", 0) > 0)

        try:
            WebDriverWait(receiver, 75, poll_frequency=2).until(is_ready)
        except Exception as exc:
            diagnostics = fetch_stats(receiver)
            (run_dir / "not_ready_diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                "No decoded inbound video was observed within 75 s. "
                "See not_ready_diagnostics.json; check CDP hook, Jitsi and camera."
            ) from exc

        if WARMUP_SECONDS:
            print(f"Video ready. {WARMUP_SECONDS} s warm-up...", flush=True)
            time.sleep(WARMUP_SECONDS)
        try:
            ok = receiver.save_screenshot(str(run_dir / "Participant_B_ui.png"))
            if not ok:
                print("WARNING: UI screenshot was not saved.", flush=True)
        except Exception as exc:
            print("WARNING: UI screenshot error:", exc, flush=True)

        print(f"{DURATION_SECONDS} s for {SAMPLE_INTERVAL_SECONDS} s interval measurement...", flush=True)
        start = time.monotonic()
        count = DURATION_SECONDS // SAMPLE_INTERVAL_SECONDS + 1
        with (run_dir / "webrtc_stats.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for index in range(count):
                wait = start + index * SAMPLE_INTERVAL_SECONDS - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elapsed = time.monotonic() - start
                row = {key: "" for key in COLUMNS}
                row.update(sample=index, elapsed_s=round(elapsed, 3),
                           timestamp_local=datetime.now().isoformat(timespec="milliseconds"))
                try:
                    data = fetch_stats(receiver)
                    (run_dir / f"raw_stats_{index:03d}.json").write_text(
                        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    stream = choose_inbound(data, selected_id)
                    video = choose_remote_video(data)
                    row["pc_count"] = data.get("pcCount", 0)
                    row["inbound_count"] = len(data.get("inboundVideos", []))
                    if not stream:
                        raise RuntimeError("No inbound-rtp video stats at this sample")
                    if selected_id is not None and stream["id"] != selected_id:
                        print("WARNING: RTP stream changed; resetting interval counters.", flush=True)
                        previous, previous_video, previous_elapsed = None, None, None
                    selected_id = stream["id"]
                    pcs = data.get("peerConnections", [])
                    pc_state = next((p.get("connectionState") for p in pcs
                                     if p["index"] == stream["pcIndex"]), "")
                    row.update({
                        "status": "OK", "pc_index": stream["pcIndex"],
                        "inbound_id": stream["id"], "ssrc": stream.get("ssrc"),
                        "codec": stream.get("codecMimeType"),
                        "connection_state": pc_state,
                        "frame_width": stream.get("frameWidth"),
                        "frame_height": stream.get("frameHeight"),
                        "reported_fps": stream.get("framesPerSecond"),
                        "frames_received": stream.get("framesReceived"),
                        "frames_decoded": stream.get("framesDecoded"),
                        "frames_dropped": stream.get("framesDropped"),
                        "packets_received": stream.get("packetsReceived"),
                        "packets_lost": stream.get("packetsLost"),
                        "bytes_received": stream.get("bytesReceived"),
                        "jitter_ms": round(stream["jitter"] * 1000, 3)
                            if number(stream.get("jitter")) is not None else None,
                        "jitter_buffer_delay_total_s": stream.get("jitterBufferDelay"),
                        "jitter_buffer_emitted_count": stream.get("jitterBufferEmittedCount"),
                        "freeze_count": stream.get("freezeCount"),
                        "total_freezes_duration_s": stream.get("totalFreezesDuration"),
                    })
                    if video:
                        row.update({
                            "remote_video_width": video.get("width"),
                            "remote_video_height": video.get("height"),
                            "playback_total_frames": video.get("totalVideoFrames"),
                            "playback_dropped_frames": video.get("droppedVideoFrames"),
                        })
                    if previous is not None and previous_elapsed is not None:
                        dt = elapsed - previous_elapsed
                        row["observed_decoded_fps"] = divide(
                            delta(stream.get("framesDecoded"), previous.get("framesDecoded")), dt)
                        row["observed_received_fps"] = divide(
                            delta(stream.get("framesReceived"), previous.get("framesReceived")), dt)
                        row["bitrate_mbps"] = divide(
                            delta(stream.get("bytesReceived"), previous.get("bytesReceived")), dt)
                        if row["bitrate_mbps"] is not None:
                            row["bitrate_mbps"] *= 8 / 1_000_000
                        lost = delta(stream.get("packetsLost"), previous.get("packetsLost"))
                        received = delta(stream.get("packetsReceived"), previous.get("packetsReceived"))
                        if lost is not None and received is not None:
                            row["interval_packet_loss_pct"] = divide(lost * 100, lost + received)
                        buffer_delay = delta(stream.get("jitterBufferDelay"), previous.get("jitterBufferDelay"))
                        emitted = delta(stream.get("jitterBufferEmittedCount"), previous.get("jitterBufferEmittedCount"))
                        if buffer_delay is not None and emitted is not None:
                            row["interval_jitter_buffer_ms_per_frame"] = divide(buffer_delay * 1000, emitted)
                        decode_time = delta(stream.get("totalDecodeTime"), previous.get("totalDecodeTime"))
                        decoded = delta(stream.get("framesDecoded"), previous.get("framesDecoded"))
                        if decode_time is not None and decoded is not None:
                            row["interval_decode_ms_per_frame"] = divide(decode_time * 1000, decoded)
                        row["interval_dropped_frames"] = delta(
                            stream.get("framesDropped"), previous.get("framesDropped"))
                        row["interval_freeze_count"] = delta(
                            stream.get("freezeCount"), previous.get("freezeCount"))
                        if video and previous_video:
                            row["observed_playback_fps"] = divide(
                                delta(video.get("totalVideoFrames"), previous_video.get("totalVideoFrames")), dt)
                    previous, previous_video, previous_elapsed = stream, video, elapsed
                    print(f"[{index:02d}] {row['frame_width']}x{row['frame_height']} "
                          f"WebRTC FPS={fmt(number(row['reported_fps']), 1)} "
                          f"measured={fmt(row['observed_decoded_fps'], 1)} "
                          f"Mbps={fmt(row['bitrate_mbps'])} "
                          f"loss%={fmt(row['interval_packet_loss_pct'])} "
                          f"jitter={fmt(number(row['jitter_ms']))}ms", flush=True)
                except Exception as exc:
                    row["status"] = "ERROR"
                    row["error"] = repr(exc)
                    print(f"[{index:02d}] HATA: {exc}", flush=True)
                    previous, previous_video, previous_elapsed = None, None, None
                writer.writerow(row)
                f.flush()
                rows.append(row)

        valid = [r for r in rows if r["status"] == "OK"]
        def average(key):
            vals = [number(r[key]) for r in valid]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        resolutions = {}
        for r in valid:
            w, h = r["frame_width"], r["frame_height"]
            key = f"{w}x{h}" if w and h else "unknown"
            resolutions[key] = resolutions.get(key, 0) + 1
        summary = {
            "samples_total": len(rows), "samples_valid": len(valid),
            "source_video": Y4M_PATH_IN_CONTAINER,
            "resolution_distribution_samples": resolutions,
            "average_reported_fps": average("reported_fps"),
            "average_observed_decoded_fps": average("observed_decoded_fps"),
            "average_observed_received_fps": average("observed_received_fps"),
            "average_observed_playback_fps": average("observed_playback_fps"),
            "average_bitrate_mbps": average("bitrate_mbps"),
            "average_interval_packet_loss_pct": average("interval_packet_loss_pct"),
            "average_jitter_ms": average("jitter_ms"),
            "average_jitter_buffer_ms_per_frame": average("interval_jitter_buffer_ms_per_frame"),
            "note": "Averages are descriptive over samples; packet-loss % is the mean of interval rates."
        }
        (run_dir / "webrtc_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nRESULTS — Participant_B receiver WebRTC statistics", flush=True)
        print(f"Measurements: {len(valid)}/{len(rows)} valid", flush=True)
        print("Resolution distribution:", resolutions, flush=True)
        print("Ortalama raporlanan FPS:", fmt(summary["average_reported_fps"]), flush=True)
        print("Mean observed decode FPS:", fmt(summary["average_observed_decoded_fps"]), flush=True)
        print("Ortalama Mbps:", fmt(summary["average_bitrate_mbps"]), flush=True)
        print("Ortalama jitter (ms):", fmt(summary["average_jitter_ms"]), flush=True)
        print("CSV:", (run_dir / "webrtc_stats.csv").resolve(), flush=True)
        print("Summary:", (run_dir / "webrtc_summary.json").resolve(), flush=True)
        if not valid:
            raise RuntimeError("No valid WebRTC samples; inspect raw_stats_*.json")
    finally:
        for driver in drivers.values():
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()

# VISUAL_ORACLE_WEBRTC_V04
