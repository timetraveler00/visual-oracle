# Visual Oracle Experiment Manager v0.2

A local English-language Streamlit dashboard for sequential, reproducible Jitsi / Selenium Grid experiments. It uses the **existing validated scripts 05, 07, 08, 09, 10 and 11_v2**, without modifying them.

## 1. Installation (Windows Anaconda Prompt)

Extract the ZIP directly into `D:\visual-oracle-revision`, replacing only the dashboard files (12_*). Keep your existing numbered source videos, Python scripts, captures, and scenario files.

```bat
conda activate visual-oracle
cd /d D:\visual-oracle-revision
python -m pip install -r requirements_dashboard.txt
```

To enable **DISTS, DINOv2, OpenCLIP**:

```bat
python -m pip install -r requirements_perceptual.txt
```

The first use of each pretrained model needs internet access to download model code/weights, unless already cached. This can take time and disk space. The application fails explicitly with an English error if a selected model cannot load; it never substitutes a fabricated metric. CPU-only mode works, but may be slow. For a quick test, deselect optional models.

Start the dashboard:

```bat
python -m streamlit run 12_visual_oracle_dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501 and leave Anaconda Prompt running. If Streamlit was already running, press Ctrl+C and restart it to pick up the updated code.

## 2. Video library

The dashboard searches **only** `D:\visual-oracle-revision\videos` (not the project root). Supported original formats: Y4M, MP4, MKV, MOV, AVI, WEBM. Create the folder if needed.

If your 665 MiB original Y4M is in the project root, copy it to `videos` or create a hard link on the same NTFS volume to avoid duplicate disk usage:

```bat
mkdir videos
mklink /H "D:\visual-oracle-revision\videos\720p50_shields_ter.y4m" "D:\visual-oracle-revision\720p50_shields_ter.y4m"
```

`mklink /H` must be run from a Windows Command Prompt on the same drive; if it fails, use File Explorer to copy the original file. Do not select a `_numbered` or `_fault_` Y4M as an original. MP4 and other compressed videos require FFmpeg on PATH; these are normalized to 1280×720 / 30 FPS / YUV420 and the configured clip length. Original Y4M files are used without conversion (must be compatible 8-bit YUV420 with at least 800×300 pixels).

## 3. Metrics and reproducibility

The dashboard always measures masked grayscale SSIM, masked RGB PSNR, DeltaE76, and WebRTC statistics. Choose optional perceptual metrics in the sidebar:

| Metric | Model / output | Interpretation |
|---|---|---|
| DISTS | DISTS-pytorch pretrained; `dists_similarity` = 1 − distance | Higher = more similar |
| DINOv2 (Base) | `facebookresearch/dinov2`, `dinov2_vitb14` | Cosine similarity of global image embeddings |
| DINOv2-Large | `facebookresearch/dinov2`, `dinov2_vitl14` | Cosine similarity of global image embeddings; slower and uses more RAM |
| OpenCLIP | `ViT-B-32`, `laion2b_s34b_b79k` | Cosine similarity of image embeddings |

The pretrained models are loaded once per dashboard job and evaluated **offline, after capture** so inference cannot affect WebRTC timing. DISTS uses 256×256 inputs; DINOv2 uses ImageNet-normalized 224×224; OpenCLIP uses its model's official image preprocessing. To exclude the overlaid frame barcode, the three perceptual metrics use an identical bottom-region crop from both reference and received frames, while SSIM/PSNR use a pixel mask. This is an important methodological distinction: **do not compare metric scores as if they used exactly the same image region**. Cosine scores are raw similarities, not percentages or calibrated probabilities.

For each run, the dashboard writes `enhanced_measurements.csv` (original `fault_measurements.csv` remains unchanged) and `perceptual_summary.json` with model details. Job-level results are in `captures/dashboard_jobs/job_*/job_summary.csv`, `status.json`, `scenario.json`, and `runner.log`. HTML/CSV reports include the additional metrics and charts. Legacy experiment records remain available.

Each scenario JSON contains selected metrics, video, conditions, durations, intervals, node capacity, and repeats. The original source Y4M and previously saved captures are not overwritten.

## 4. Node capacity

Configure 2–10 existing Selenium Grid Docker node containers. Each run uses **two participants**, sender and receiver; conditions and repetitions run sequentially. The sender's source Y4M is copied only to its assigned node, using the proven Grid session-to-container resolution logic.

## 5. Limitations

- Full end-to-end operation was not tested in your Windows/Docker environment; verify first with one baseline run.
- First-time model downloads require connectivity; model load errors stop the job before opening Jitsi.
- Perceptual metrics can be slow on CPU. Allow time after the WebRTC capture completes.
- Session-to-session resolution differences, variable source content and sparse samples can confound metric comparisons.
- The dashboard presents English text and translates known Turkish progress messages emitted by the unchanged legacy scripts. Unexpected legacy exception text may still be Turkish; full localization of the old scripts would require modifying those scripts.
- Node capacity is not parallelism, and one 60-second session's frames are not independent experimental repetitions.
