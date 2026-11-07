GOOGLE MEET AUTOMATED CAPTURE - PROTOTYPE
=========================================
1) Extract config.json and meet_auto.py to D:\visual-oracle-revision\
2) Edit config.json: set meet_url to a real, pre-created Meet URL.
3) In Anaconda Prompt:
   conda activate visual-oracle
   python -m pip install selenium opencv-python scikit-image numpy
   cd /d D:\visual-oracle-revision\docker
   docker compose up -d
   docker compose ps
   python D:\visual-oracle-revision\meet_auto.py

Requires one Selenium Chrome node per participant. Existing Docker Compose
configuration has 2 Chrome nodes; use participants=2.

Meeting admission and Google sign-in restrictions are not bypassed. A meeting
configured to allow these test browsers to join is required. If joining fails,
inspect the join_failed.png file in the timestamped output directory.

The script automatically opens browsers, attempts to join, captures PNGs at a
specified interval, transfers screenshots via Selenium to the controller, and
writes CSV with SSIM to first/previous frames. This is NOT yet a validated
video-quality or freeze-detection oracle. SSIM is diagnostic only.

Do not place account passwords in config.json. Only capture consented test meetings.
