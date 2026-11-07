"""Jitsi Meet: Y4M camera on Participant_A, Selenium Grid two-person test."""
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

GRID_URL = "http://localhost:4444"
JITSI_URL = "https://host.docker.internal:8443/VisualOracleLab"
DURATION_SECONDS = 120
Y4M_PATH_IN_CONTAINER = "/tmp/720p50_shields_ter.y4m"
OUTPUT_DIR = Path("screenshots") / "y4m_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def conference_state(driver):
    return driver.execute_script("""
        const conf = window.APP && window.APP.conference;
        const room = conf && conf._room;
        let joined = null;
        let remoteParticipants = null;
        try {
          if (room && typeof room.isJoined === 'function')
            joined = Boolean(room.isJoined());
          else if (conf && typeof conf.isJoined === 'function')
            joined = Boolean(conf.isJoined());
        } catch (e) {}
        try {
          if (room && typeof room.getParticipants === 'function')
            remoteParticipants = room.getParticipants().length;
        } catch (e) {}
        const videos = Array.from(document.querySelectorAll('video'));
        const playingVideos = videos.filter(v => v.readyState >= 2 &&
          v.videoWidth > 0 && v.videoHeight > 0).length;
        const remoteVideos = videos.filter(v =>
          /remote/i.test(v.id || '') && v.readyState >= 2 &&
          v.videoWidth > 0 && v.videoHeight > 0).length;
        return { joined, remoteParticipants, playingVideos,
                 remoteVideos, totalVideos: videos.length };
    """)


def start_participant(name):
    options = Options()
    for flag in (
        "--ignore-certificate-errors",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,720",
    ):
        options.add_argument(flag)
    if name == "Participant_A":
        options.add_argument(
            f"--use-file-for-fake-video-capture={Y4M_PATH_IN_CONTAINER}"
        )
    options.page_load_strategy = "eager"

    driver = webdriver.Remote(command_executor=GRID_URL, options=options)
    try:
        driver.get(JITSI_URL)
        name_input = WebDriverWait(driver, 40).until(find_name_input)
        name_input.clear()
        name_input.send_keys(name)
        join_button = WebDriverWait(driver, 15).until(find_join_button)
        join_button.click()
        print(f"{name}: Clicked Join meeting.", flush=True)

        # Wait until the pre-join UI goes away. This is not proof of media connectivity.
        WebDriverWait(driver, 50).until(lambda d: find_join_button(d) is None)
        print(f"{name}: Left the pre-join screen.", flush=True)
        return driver
    except Exception:
        driver.save_screenshot(str(OUTPUT_DIR / f"{name}_error.png"))
        driver.quit()
        raise


def main():
    drivers = {}
    try:
        for name in ("Participant_A", "Participant_B"):
            print(f"{name}: Starting...", flush=True)
            drivers[name] = start_participant(name)

        for minute in range(DURATION_SECONDS // 60 + 1):
            if minute:
                time.sleep(60)
            print(f"\n[{datetime.now():%H:%M:%S}] Dakika {minute}", flush=True)
            for name, driver in drivers.items():
                try:
                    state = conference_state(driver)
                    screenshot = OUTPUT_DIR / f"{name}_minute_{minute}.png"
                    driver.save_screenshot(str(screenshot))
                    print(f"{name}: {state} | {screenshot}", flush=True)
                except Exception as exc:
                    print(f"{name}: Monitoring error: {exc}", flush=True)
        print("\nTwo-minute Y4M test completed.", flush=True)
        print("NOTE: Check screenshots to confirm Participant_A shows the harbour video rather than the green test pattern.")
    finally:
        for driver in drivers.values():
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
