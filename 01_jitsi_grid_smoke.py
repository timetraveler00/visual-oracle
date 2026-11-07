
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

GRID_URL = "http://localhost:4444"
JITSI_URL = "https://host.docker.internal:8443/VisualOracleLab"

output = Path("screenshots")
output.mkdir(exist_ok=True)

drivers = []

try:
    for i in range(1, 3):
        options = Options()
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--use-fake-ui-for-media-stream")
        options.add_argument("--use-fake-device-for-media-stream")
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--window-size=1280,720")
        options.page_load_strategy = "eager"

        driver = webdriver.Remote(
            command_executor=GRID_URL,
            options=options
        )

        drivers.append(driver)
        print(f"Chrome {i}: Session created")

        driver.get(JITSI_URL)
        print(f"Chrome {i}: Page opened")

    time.sleep(15)

    for i, driver in enumerate(drivers, start=1):
        print(f"Chrome {i} title: {driver.title}")
        print(f"Chrome {i} adres: {driver.current_url}")

        filename = output / f"chrome_{i}.png"
        driver.save_screenshot(str(filename))
        print(f"Screenshot: {filename}")

    print("Initial test completed.")

finally:
    for driver in drivers:
        driver.quit()
