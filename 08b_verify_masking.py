"""Verify 08's masked SSIM/PSNR and report quality grouped by received resolution.

Windows / Anaconda Prompt:
  cd /d D:\\visual-oracle-revision
  python 08b_verify_masking.py
  python 08b_verify_masking.py --run captures\\shields_20261101_110029

Requires original 08_jitsi_synchronized_quality.py and its input files.
Does NOT modify source, PNG captures, or existing results.
"""
import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_module():
    path = Path(__file__).with_name('08_jitsi_synchronized_quality.py')
    if not path.is_file():
        raise FileNotFoundError(f'08 script not found: {path}')
    spec = importlib.util.spec_from_file_location('jitsi_quality_08', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def almost_equal(a, b, tol=1e-9):
    if math.isinf(a) and math.isinf(b):
        return True
    return abs(a - b) <= tol


def synthetic_test(m):
    """Independent sanity checks: masked modifications ignored; visible ones detected."""
    rng = np.random.default_rng(1234)
    a = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    b = a.copy()
    mask = np.ones((180, 320), dtype=bool)
    # Excluded 70x35 area + 8px safety margin (for 11x11 SSIM support).
    mask[2:53, 2:88] = False
    ss0, p0 = m.quality_metrics(a, b, mask)
    b[10:45, 10:80] = 255 - b[10:45, 10:80]
    ss1, p1 = m.quality_metrics(a, b, mask)
    assert almost_equal(ss0, ss1) and almost_equal(p0, p1), 'Masked modification changed score'
    b[80:140, 130:230] = 255 - b[80:140, 130:230]
    ss2, p2 = m.quality_metrics(a, b, mask)
    assert ss2 < ss1 - 0.001 and p2 < p1, 'Visible modification was not detected'
    print('Synthetic test: PASS (excluded pixels ignored; visible pixels detected)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='Existing numbered capture folder')
    args = ap.parse_args()
    m = load_module()
    synthetic_test(m)
    metadata = json.loads(m.METADATA.read_text(encoding='utf-8-sig'))
    run = m.find_run(args.run)
    w, h, fps, offsets = m.index_y4m(m.ORIGINAL)
    files = sorted(run.glob('Participant_B_remote_*.png'))
    if not files:
        raise FileNotFoundError('No Participant_B_remote_*.png files')
    print('Capture folder:', run.resolve())
    print('Reference:', m.ORIGINAL.resolve())
    print(f'Source: {w}x{h}, {fps:.3f} fps, {len(offsets)} frames')
    old = {}
    old_path = run / 'quality_synchronized.csv'
    if old_path.exists():
        with old_path.open(newline='', encoding='utf-8-sig') as fh:
            old = {r['filename']: r for r in csv.DictReader(fh)}
    output_rows = []
    groups = defaultdict(list)
    failures = 0
    with m.ORIGINAL.open('rb') as handle:
        for path in files:
            rec = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if rec is None:
                print(f'FAIL {path.name}: PNG unreadable')
                failures += 1
                continue
            rh, rw = rec.shape[:2]
            try:
                idx, confidence, code = m.decode_barcode(rec, metadata)
                if idx >= len(offsets):
                    raise ValueError(f'Frame index outside source: {idx}')
                ref = m.read_reference(handle, offsets[idx], w, h, rw, rh)
                mask, rect = m.valid_pixel_mask(rec.shape, metadata)
                ssim, psnr = m.quality_metrics(ref, rec, mask)
                # Change ONLY the original overlay core, safely inside excluded area.
                x0, y0, x1, y1 = metadata['overlay_rect_xyxy_source_pixels']
                xa = max(0, math.ceil(x0 * rw / w))
                ya = max(0, math.ceil(y0 * rh / h))
                xb = min(rw, math.floor(x1 * rw / w))
                yb = min(rh, math.floor(y1 * rh / h))
                assert xb > xa and yb > ya
                assert not mask[ya:yb, xa:xb].any(), 'Overlay core not excluded!'
                rec_changed = rec.copy()
                rec_changed[ya:yb, xa:xb] = 255 - rec_changed[ya:yb, xa:xb]
                ss_mod, ps_mod = m.quality_metrics(ref, rec_changed, mask)
                overlay_invariant = almost_equal(ssim, ss_mod, 1e-8) and almost_equal(psnr, ps_mod, 1e-8)
                # Confirm visible changes alter both metrics.
                visible = np.argwhere(mask)
                cy, cx = visible[len(visible) // 2]
                # Change an unmasked 12x12 patch well away from overlay.
                cx = min(rw-25, max(25, rw//2))
                cy = min(rh-25, max(25, rh//2))
                rec_visible = rec.copy()
                rec_visible[cy:cy+20, cx:cx+20] = 255 - rec_visible[cy:cy+20, cx:cx+20]
                ss_visible, ps_visible = m.quality_metrics(ref, rec_visible, mask)
                visible_sensitive = abs(ssim - ss_visible) > 1e-7 and abs(psnr - ps_visible) > 1e-7
                # Compare with incorrect approach (black-fill both images, unmasked SSIM).
                ref_black, rec_black = ref.copy(), rec.copy()
                rx0, ry0, rx1, ry1 = rect
                ref_black[ry0:ry1, rx0:rx1] = 0
                rec_black[ry0:ry1, rx0:rx1] = 0
                all_valid = np.ones_like(mask)
                ss_black, ps_black = m.quality_metrics(ref_black, rec_black, all_valid)
                prev = old.get(path.name, {})
                prior_s = prev.get('ssim_gray_masked', '')
                prior_p = prev.get('psnr_rgb_masked_db', '')
                previous_matches = (not prior_s or abs(float(prior_s)-ssim) < 1e-5) and (not prior_p or abs(float(prior_p)-psnr) < 0.00011)
                ok = overlay_invariant and visible_sensitive and previous_matches
                if not ok:
                    failures += 1
                groups[(rw, rh)].append((ssim, psnr))
                output_rows.append({
                    'filename': path.name, 'frame_index': idx, 'width': rw, 'height': rh,
                    'valid_percent': round(100*float(mask.mean()), 3),
                    'masked_ssim': round(ssim, 6), 'masked_psnr_db': round(psnr, 4),
                    'black_fill_ssim_wrong_method': round(ss_black, 6),
                    'black_fill_psnr_wrong_method_db': round(ps_black, 4),
                    'overlay_invariant': overlay_invariant,
                    'visible_sensitive': visible_sensitive,
                    'previous_result_matches': previous_matches,
                    'verification': 'PASS' if ok else 'FAIL',
                })
                print(f'{path.name}: FRAME {idx:06d} | {rw}x{rh} | '
                      f'valid={100*mask.mean():.1f}% | masked SSIM={ssim:.4f} | '
                      f'black-fill SSIM={ss_black:.4f} | {"PASS" if ok else "FAIL"}')
            except Exception as exc:
                failures += 1
                print(f'FAIL {path.name}: {exc}')
    out = run / 'mask_validation.csv'
    if output_rows:
        with out.open('w', newline='', encoding='utf-8-sig') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    print('\nResolution-grouped quality (not equivalent sample sizes):')
    for (rw, rh), vals in sorted(groups.items()):
        print(f'  {rw}x{rh}: n={len(vals)}, SSIM={np.mean([v[0] for v in vals]):.4f}, '
              f'PSNR={np.mean([v[1] for v in vals]):.2f} dB')
    print(f'\nVerification: {len(output_rows)-sum(r["verification"] == "FAIL" for r in output_rows)}/{len(files)} passed; failures={failures}')
    print('CSV:', out.resolve())
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
