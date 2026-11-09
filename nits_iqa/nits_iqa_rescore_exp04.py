#!/usr/bin/env python3
"""Re-score the 405 NITS-IQA pairs with the frozen EXP04 model settings.

Same models and preprocessing as exp04_deep_calibration.py:
  DISTS      DISTS_pytorch pretrained, RGB in [0,1], native distance
  DINOv2     facebookresearch/dinov2:dinov2_vitb14 (and vitl14), CLS cosine
  OpenCLIP   ViT-B-32, laion2b_s34b_b79k, image-embedding cosine
  Deep input whole image letterboxed to 224x224 (bicubic, RGB 127 padding), no crop
  SSIM       grayscale, 11x11 Gaussian window, sigma 1.5 (same formula as EXP04)
  PSNR       RGB
NITS-IQA images are scored at native resolution (no ROI mask).

Usage (from the repository root):
  python nits_iqa_rescore_exp04.py --images PATH_TO_NITS_IQA_IMAGES
Output: nits_iqa/verified/NITS_IQA_405_rescored_exp04.csv
The output keeps the original column names, so it can be analysed with
  python nits_iqa/verified/nits_iqa_revision_analysis.py nits_iqa/verified/NITS_IQA_405_rescored_exp04.csv
"""
import argparse, math, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps

NEUTRAL = (127, 127, 127)

def find(folder, name):
    hits = list(Path(folder).rglob(name))
    if not hits:
        hits = [p for p in Path(folder).rglob("*") if p.name.lower() == name.lower()]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {folder}")
    return hits[0]

def ssim_exp04(ref_bgr, rec_bgr):
    x = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mx, my = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    sx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
    sy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
    sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
    c1, c2 = (.01 * 255) ** 2, (.03 * 255) ** 2
    smap = ((2 * mx * my + c1) * (2 * sxy + c2)) / np.maximum((mx ** 2 + my ** 2 + c1) * (sx + sy + c2), 1e-9)
    return float(smap.mean())

def psnr(ref_bgr, rec_bgr):
    mse = float(np.mean((ref_bgr.astype(np.float64) - rec_bgr.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder containing the NITS-IQA reference and distorted images")
    ap.add_argument("--table", default="nits_iqa/verified/NITS_IQA_405_verified.csv")
    ap.add_argument("--out", default="nits_iqa/verified/NITS_IQA_405_rescored_exp04.csv")
    ap.add_argument("--no-large", action="store_true", help="skip DINOv2-Large")
    a = ap.parse_args()

    import torch, torch.nn.functional as F, open_clip
    from DISTS_pytorch import DISTS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}", flush=True)
    dists = DISTS().eval().to(dev)
    dino_b = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
    dino_l = None if a.no_large else torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True).eval().to(dev)
    clip, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    clip.eval()
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1); std = torch.tensor([.229, .224, .225]).view(3, 1, 1)

    def tensors(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = ImageOps.pad(Image.fromarray(rgb), (224, 224), method=Image.Resampling.BICUBIC, color=NEUTRAL, centering=(.5, .5))
        t_d = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255)
        t_dino = (torch.from_numpy(np.array(small).transpose(2, 0, 1).copy()).float() / 255 - mean) / std
        return t_d, t_dino, clip_pre(small)

    def cos(model, x, y, enc=lambda m, v: m(v)):
        e = F.normalize(enc(model, torch.stack([x, y]).to(dev)).float(), dim=-1)
        return float((e[0] * e[1]).sum())

    tab = pd.read_csv(a.table)
    # Local (non-published) copies may use a different header, e.g. a Turkish
    # name for the distorted-image column. Map the first two columns by position.
    cols = list(tab.columns)
    if "Ref" not in cols or "Distorted" not in cols:
        print(f"[table] header {cols[:2]} mapped to ['Ref', 'Distorted']", flush=True)
        tab = tab.rename(columns={cols[0]: "Ref", cols[1]: "Distorted"})
    if not str(tab["Distorted"].iloc[0]).upper().startswith("I") or "D" not in str(tab["Distorted"].iloc[0]).upper():
        sys.exit(f"Unexpected table layout; first row: {tab.iloc[0].to_dict()}")
    rows = []
    with torch.inference_mode():
        for i, r in tab.iterrows():
            ref = cv2.imread(str(find(a.images, r["Ref"])), cv2.IMREAD_COLOR)
            dis = cv2.imread(str(find(a.images, r["Distorted"])), cv2.IMREAD_COLOR)
            if ref is None or dis is None or ref.shape != dis.shape:
                sys.exit(f"Cannot read or size mismatch: {r['Ref']} / {r['Distorted']}")
            (rd, rdi, rc), (dd, ddi, dc) = tensors(ref), tensors(dis)
            dist = float(dists(rd[None].to(dev), dd[None].to(dev), require_grad=False, batch_average=False).reshape(-1)[0])
            out = {"Ref": r["Ref"], "Distorted": r["Distorted"],
                   "DINOv2-Base (%)": 100 * cos(dino_b, rdi, ddi),
                   "DINOv2-Large (%)": 100 * cos(dino_l, rdi, ddi) if dino_l is not None else np.nan,
                   "Dists (%)": 100 * (1 - dist),
                   "SSIM (%)": 100 * ssim_exp04(ref, dis),
                   "OpenCLIP (%)": 100 * cos(clip, rc, dc, lambda m, v: m.encode_image(v)),
                   "MOS": r["MOS"], "DISTS_distance": dist, "PSNR_dB": psnr(ref, dis)}
            rows.append(out)
            if (i + 1) % 45 == 0:
                print(f"{i + 1}/405", flush=True)
    new = pd.DataFrame(rows)
    new.to_csv(a.out, index=False, float_format="%.4f")
    print(f"[saved] {a.out}")

    # agreement with the historical score table
    from scipy.stats import spearmanr
    print("\nAgreement with the historical scores (Spearman across 405 pairs):")
    if not any(c in tab.columns for c in ["DINOv2-Base (%)", "Dists (%)", "SSIM (%)", "OpenCLIP (%)"]):
        print("  metric columns not found in this table; rerun with --table nits_iqa/verified/NITS_IQA_405_verified.csv")
    for c in ["DINOv2-Base (%)", "DINOv2-Large (%)", "Dists (%)", "SSIM (%)", "OpenCLIP (%)"]:
        if c in tab.columns and new[c].notna().all():
            print(f"  {c:18s} rho = {spearmanr(tab[c], new[c])[0]:.4f}   "
                  f"mean old {tab[c].mean():6.2f}  new {new[c].mean():6.2f}")

if __name__ == "__main__":
    main()
