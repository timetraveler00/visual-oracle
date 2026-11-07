
"""Jitsi/Selenium Grid: capture actual remote <video> frames at Participant_B.

Run in Windows Anaconda Prompt: python 05_jitsi_video_capture.py
Prerequisite: /tmp/720p50_shields_ter.y4m exists in BOTH Selenium Chrome nodes.
Requires selenium. Does NOT compute SSIM or prove received WebRTC FPS.
"""

import base64
import csv
import json
import struct
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

GRID_URL = "http://localhost:4444"
JITSI_URL = "https://host.docker.internal:8443/VisualOracleLab"
Y4M_PATH_IN_CONTAINER = "/tmp/720p50_shields_ter_numbered.y4m"
DURATION_SECONDS = 60
CAPTURE_INTERVAL_SECONDS = 10
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

# Only capture video elements identified as REMOTE by Jitsi's DOM.
# No fallback to a local video: silently capturing local would invalidate the experiment.
CAPTURE_REMOTE_JS = r"""
const videos = Array.from(document.querySelectorAll('video'));
const candidates = videos.map((v, i) => {
  const parent = v.closest('.videocontainer, .remote-video-container');
  const id = v.id || '';
  const cls = (typeof v.className === 'string') ? v.className : '';
  const parentId = parent ? (parent.id || '') : '';
  const parentCls = parent ? (parent.className || '') : '';
  const remote = /remote/i.test(id) || /remote/i.test(cls) ||
                 /remote/i.test(parentId) || /remote/i.test(parentCls) ||
                 /^participant_/.test(parentId);
  const local = /local/i.test(id) || /local/i.test(cls) ||
                /local/i.test(parentId) || /local/i.test(parentCls);
  return {v, i, id, cls, parentId, remote, local,
          readyState: v.readyState, w: v.videoWidth, h: v.videoHeight};
});
const valid = candidates.filter(x => x.remote && !x.local &&
   x.readyState >= 2 && x.w > 0 && x.h > 0);
valid.sort((a,b) => (b.w*b.h) - (a.w*a.h));
if (!valid.length) {
  return {ok:false, error:'No ready remote video found',
    candidates:candidates.map(x => ({id:x.id, cls:x.cls,
      parentId:x.parentId, remote:x.remote, local:x.local,
      readyState:x.readyState, width:x.w, height:x.h}))};
}
const x = valid[0];
const v = x.v;
const canvas = document.createElement('canvas');
canvas.width = v.videoWidth;
canvas.height = v.videoHeight;
const ctx = canvas.getContext('2d', {willReadFrequently:true});
try {
  ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
  // Sample pixels to help detect all-black/blank captures; not a quality metric.
  const samples = [];
  for (let yy=1; yy<=4; yy++) for (let xx=1; xx<=4; xx++) {
    const px = ctx.getImageData(Math.floor(canvas.width*xx/5),
                                Math.floor(canvas.height*yy/5),1,1).data;
    samples.push(Math.round((px[0]+px[1]+px[2])/3));
  }
  const avg = samples.reduce((a,b)=>a+b,0)/samples.length;
  const variance = samples.reduce((a,b)=>a+(b-avg)*(b-avg),0)/samples.length;
  const quality = typeof v.getVideoPlaybackQuality === 'function' ?
                  v.getVideoPlaybackQuality() : null;
  return {ok:true, dataUrl:canvas.toDataURL('image/png'),
    elementId:x.id, parentId:x.parentId,
    width:canvas.width, height:canvas.height,
    videoCurrentTime:v.currentTime,
    webkitDecodedFrameCount:v.webkitDecodedFrameCount ?? null,
    totalVideoFrames:quality ? quality.totalVideoFrames : null,
    droppedVideoFrames:quality ? quality.droppedVideoFrames : null,
    pixelSampleMean:Math.round(avg*100)/100,
    pixelSampleVariance:Math.round(variance*100)/100,
    videoElements:videos.length,
    capturedAtBrowser:new Date().toISOString()};
} catch(e) {
  return {ok:false, error:String(e), elementId:x.id,
          width:x.w, height:x.h};
}
"""


def find_name_input(driver):
    for selector in NAME_SELECTORS:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            if element.is_displayed() and element.is_enabled():
                return element
    return None


def find_join_button(driver):
    for element in driver.find_elements(By.XPATH, JOIN_XPATH):
        if element.is_displayed() and element.is_enabled():
            return element
    return None


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
        options.add_argument(
            f"--use-file-for-fake-video-capture={Y4M_PATH_IN_CONTAINER}"
        )
    options.page_load_strategy = "eager"
    driver = webdriver.Remote(command_executor=GRID_URL, options=options)
    driver.set_script_timeout(30)
    try:
        driver.get(JITSI_URL)
        name_input = WebDriverWait(driver, 40).until(find_name_input)
        name_input.clear()
        name_input.send_keys(name)
        join_button = WebDriverWait(driver, 15).until(find_join_button)
        join_button.click()
        WebDriverWait(driver, 50).until(lambda d: find_join_button(d) is None)
        print(f"{name}: joined the meeting (UI).", flush=True)
        return driver
    except Exception:
        driver.quit()
        raise


def capture_frame(driver, target):
    result = driver.execute_script(CAPTURE_REMOTE_JS)
    if not result or not result.get("ok"):
        return False, result or {"error": "Empty JavaScript response"}
    prefix = "data:image/png;base64,"
    encoded = result.pop("dataUrl", "")
    if not encoded.startswith(prefix):
        return False, {"error": "PNG data URL bekleniyordu", **result}
    payload = base64.b64decode(encoded[len(prefix):], validate=True)
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return False, {"error": "Invalid PNG header", **result}
    width, height = struct.unpack(">II", payload[16:24])
    if width != result["width"] or height != result["height"]:
        return False, {"error": "PNG dimensions do not match", **result}
    target.write_bytes(payload)
    if target.stat().st_size != len(payload):
        return False, {"error": "File size mismatch", **result}
    result["fileBytes"] = len(payload)
    return True, result


def main():
    # Unique output directory prevents stale captures being mistaken for new ones.
    run_dir = OUTPUT_ROOT / datetime.now().strftime("shields_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print("Output directory:", run_dir.resolve(), flush=True)
    (run_dir / "settings.json").write_text(json.dumps({
        "grid": GRID_URL, "jitsi": JITSI_URL,
        "source_video": Y4M_PATH_IN_CONTAINER,
        "duration_seconds": DURATION_SECONDS,
        "capture_interval_seconds": CAPTURE_INTERVAL_SECONDS,
        "capture_method": "remote HTMLVideoElement -> canvas -> PNG",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    drivers = {}
    columns = ["sample", "timestamp_local", "status", "filename", "error",
               "width", "height", "elementId", "parentId", "videoCurrentTime",
               "webkitDecodedFrameCount", "totalVideoFrames", "droppedVideoFrames",
               "pixelSampleMean", "pixelSampleVariance", "fileBytes", "capturedAtBrowser"]
    try:
        for name in ("Participant_A", "Participant_B"):
            print(name, "starting...", flush=True)
            drivers[name] = start_participant(name)
        b = drivers["Participant_B"]
        print("Waiting for remote video (up to 60 seconds)...", flush=True)
        WebDriverWait(b, 60, poll_frequency=2).until(
            lambda d: bool(d.execute_script("""
              return [...document.querySelectorAll('video')].some(v =>
                /remote/i.test(v.id || '') && v.readyState >= 2 &&
                v.videoWidth > 0 && v.videoHeight > 0);
            """))
        )
        # Save the UI separately for diagnostics; never use as a video frame.
        screenshot = run_dir / "Participant_B_ui.png"
        ok = b.save_screenshot(str(screenshot))
        print("UI screenshot:", screenshot if ok and screenshot.exists() else "FAILED", flush=True)
        with (run_dir / "capture_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            start = time.monotonic()
            count = DURATION_SECONDS // CAPTURE_INTERVAL_SECONDS + 1
            for index in range(count):
                wait = start + index * CAPTURE_INTERVAL_SECONDS - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                target = run_dir / f"Participant_B_remote_{index:03d}.png"
                timestamp = datetime.now().isoformat(timespec="milliseconds")
                try:
                    success, data = capture_frame(b, target)
                except Exception as exc:
                    success, data = False, {"error": repr(exc)}
                row = {"sample": index, "timestamp_local": timestamp,
                       "status": "OK" if success else "ERROR",
                       "filename": target.name if success else "",
                       "error": "" if success else data.get("error", "unknown")}
                row.update({k: data.get(k, "") for k in columns if k not in row})
                writer.writerow(row)
                f.flush()
                if success:
                    print(f"[{index:02d}] OK {target.name} "
                          f"{data['width']}x{data['height']} "
                          f"({data['fileBytes']} bytes), "
                          f"pixel variance={data['pixelSampleVariance']}", flush=True)
                    if data["pixelSampleVariance"] < 1:
                        print("  WARNING: Frame may be uniform/black; inspect the PNG.", flush=True)
                else:
                    print(f"[{index:02d}] HATA: {row['error']}", flush=True)
                    (run_dir / f"debug_{index:03d}.json").write_text(
                        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nFinished. PNG files and capture_log.csv:", run_dir.resolve(), flush=True)
        print("NOTE: This stage captures video frames; reference alignment/SSIM is a later stage.")
    finally:
        for driver in drivers.values():
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
