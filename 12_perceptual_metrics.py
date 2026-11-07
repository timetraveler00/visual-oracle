"""Optional offline perceptual metrics for Visual Oracle Experiment Manager.

The overlay is EXCLUDED by cropping the identical bottom rectangle from both
images. DISTS is a similarity (1 - published DISTS distance); DINOv2 and
OpenCLIP values are raw cosine similarities, not probabilities or percentages.
Model weights are downloaded on first use and cached by PyTorch/OpenCLIP.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

METRIC_COLUMNS = {'DISTS': 'dists_similarity', 'DINOv2': 'dinov2_cosine',
                  'OpenCLIP': 'openclip_cosine',
                  'DINOv2-Large': 'dinov2_large_cosine'}


def import_local(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerceptualEngine:
    def __init__(self, selected, device='auto'):
        self.selected = list(dict.fromkeys(selected))
        unknown = set(self.selected) - set(METRIC_COLUMNS)
        if unknown:
            raise ValueError(f'Unknown perceptual metrics: {sorted(unknown)}')
        self.models = {}
        self.device = None
        self.torch = None
        self.preprocess_clip = None
        if not self.selected:
            return
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError('Perceptual metrics require PyTorch and Pillow. Install requirements_perceptual.txt') from exc
        self.torch, self.F, self.Image = torch, F, Image
        self.device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else ('cpu' if device == 'auto' else device)
        if self.device == 'cpu':
            torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        print(f'[Metrics] Device: {self.device}; loading {", ".join(self.selected)}', flush=True)
        if 'DISTS' in self.selected:
            try:
                from DISTS_pytorch import DISTS
            except ImportError as exc:
                raise RuntimeError('DISTS dependency missing: python -m pip install DISTS-pytorch') from exc
            self.models['DISTS'] = DISTS().to(self.device).eval()
            print('[Metrics] DISTS model ready', flush=True)
        if 'DINOv2' in self.selected:
            try:
                self.models['DINOv2'] = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True,
                    trust_repo=True, skip_validation=True).to(self.device).eval()
            except Exception as exc:
                raise RuntimeError('DINOv2 weights unavailable. First use requires access to the official facebookresearch/dinov2 repository and pretrained weights, or an existing Torch Hub cache. Original error: ' + str(exc)) from exc
            print('[Metrics] DINOv2 ViT-B/14 ready', flush=True)
        if 'DINOv2-Large' in self.selected:
            try:
                self.models['DINOv2-Large'] = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True,
                    trust_repo=True, skip_validation=True).to(self.device).eval()
            except Exception as exc:
                raise RuntimeError('DINOv2-Large weights unavailable. First use requires access to the official facebookresearch/dinov2 repository and pretrained weights, or an existing Torch Hub cache. Original error: ' + str(exc)) from exc
            print('[Metrics] DINOv2 ViT-L/14 ready', flush=True)
        if 'OpenCLIP' in self.selected:
            try:
                import open_clip
            except ImportError as exc:
                raise RuntimeError('OpenCLIP dependency missing: python -m pip install open_clip_torch') from exc
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device)
            except Exception as exc:
                raise RuntimeError('OpenCLIP pretrained weights unavailable. First use needs internet or cached weights. Original error: ' + str(exc)) from exc
            self.models['OpenCLIP'] = model.eval()
            self.preprocess_clip = preprocess
            print('[Metrics] OpenCLIP ViT-B-32 (LAION-2B) ready', flush=True)

    def _rgb_pil(self, bgr):
        return self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _tensor(self, pil, size, imagenet=False):
        img = pil.resize((size, size), resample=self.Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32).copy() / 255.0
        t = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if imagenet:
            mean = t.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = t.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            t = (t - mean) / std
        return t

    def compare(self, reference_bgr, received_bgr, overlay_bottom):
        """Compare the same barcode-free lower crop of aligned BGR frames."""
        if reference_bgr.shape != received_bgr.shape:
            raise ValueError('Reference and received frame shapes differ')
        height = reference_bgr.shape[0]
        y0 = max(0, min(height - 8, int(overlay_bottom) + max(6, round(height * .01))))
        ref = self._rgb_pil(reference_bgr[y0:])
        rec = self._rgb_pil(received_bgr[y0:])
        out = {}
        with self.torch.inference_mode():
            if 'DISTS' in self.models:
                a, b = self._tensor(ref, 256), self._tensor(rec, 256)
                dist = float(self.models['DISTS'](a, b).item())
                out['dists_similarity'] = 1.0 - dist
            if 'DINOv2' in self.models:
                a, b = self._tensor(ref, 224, imagenet=True), self._tensor(rec, 224, imagenet=True)
                model = self.models['DINOv2']
                emb_a, emb_b = model(a), model(b)
                out['dinov2_cosine'] = float(self.F.cosine_similarity(emb_a.flatten(1), emb_b.flatten(1)).item())
            if 'DINOv2-Large' in self.models:
                a, b = self._tensor(ref, 224, imagenet=True), self._tensor(rec, 224, imagenet=True)
                model = self.models['DINOv2-Large']
                emb_a, emb_b = model(a), model(b)
                out['dinov2_large_cosine'] = float(self.F.cosine_similarity(emb_a.flatten(1), emb_b.flatten(1)).item())
            if 'OpenCLIP' in self.models:
                a = self.preprocess_clip(ref).unsqueeze(0).to(self.device)
                b = self.preprocess_clip(rec).unsqueeze(0).to(self.device)
                model = self.models['OpenCLIP']
                emb_a, emb_b = model.encode_image(a), model.encode_image(b)
                out['openclip_cosine'] = float(self.F.cosine_similarity(emb_a.float(), emb_b.float()).item())
        return out


def average(rows, key, active_only=False):
    vals = []
    for row in rows:
        if active_only and str(row.get('fault_active')) not in ('1', '1.0', 'True'):
            continue
        try:
            x = float(row.get(key, ''))
            if math.isfinite(x):
                vals.append(x)
        except (TypeError, ValueError):
            pass
    return round(float(np.mean(vals)), 6) if vals else None


def enrich_run(run_folder, source, metadata_path, engine, root):
    """Preserve original fault_measurements.csv and add enhanced_measurements.csv."""
    if not engine.selected:
        return {}
    run_folder = Path(run_folder)
    source = Path(source)
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8-sig'))
    quality = import_local(Path(root) / '08_jitsi_synchronized_quality.py', 'quality08_metrics')
    original_csv = run_folder / 'fault_measurements.csv'
    if not original_csv.is_file():
        raise FileNotFoundError(f'Missing receiver measurements: {original_csv}')
    with original_csv.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    w, h, fps, offsets = quality.index_y4m(source)
    if len(offsets) != int(metadata['frame_count']):
        raise ValueError('Reference video and metadata frame counts do not match')
    counts = {name: 0 for name in engine.selected}
    with source.open('rb') as ref_file:
        for i, row in enumerate(rows):
            for col in METRIC_COLUMNS.values():
                row[col] = ''
            if row.get('quality_status') != 'OK':
                continue
            try:
                frame_idx = int(row['frame_index'])
                png_path = run_folder / Path(row['capture_filename']).name
                if frame_idx < 0 or frame_idx >= len(offsets) or not png_path.is_file():
                    raise ValueError('Invalid frame index or missing PNG')
                received = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
                if received is None:
                    raise ValueError('PNG could not be decoded')
                rh, rw = received.shape[:2]
                reference = quality.read_reference(ref_file, offsets[frame_idx], w, h, rw, rh)
                _, (x0, y0, x1, y1) = quality.valid_pixel_mask(received.shape, metadata)
                values = engine.compare(reference, received, y1)
                for key, value in values.items():
                    row[key] = round(value, 6)
                for metric in engine.selected:
                    counts[metric] += 1
                print(f'[Metrics] Frame {i+1}/{len(rows)}: ' + ', '.join(f'{key}={value:.4f}' for key, value in values.items()), flush=True)
            except Exception as exc:
                raise RuntimeError(f'Perceptual metric computation failed at sample {i}, frame {row.get("frame_index")}: {exc}') from exc
    columns = list(rows[0]) if rows else []
    output = run_folder / 'enhanced_measurements.csv'
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'metrics': engine.selected, 'device': engine.device, 'sample_counts': counts,
               'preprocessing': 'identical overlay-free bottom crop; resize 256x256 for DISTS, 224x224 for DINOv2, official OpenCLIP preprocessing',
               'metric_definitions': {'dists_similarity': '1 - DISTS distance (higher = more similar)',
                                      'dinov2_cosine': 'cosine similarity of DINOv2 ViT-B/14 global embeddings',
                                      'openclip_cosine': 'cosine similarity of OpenCLIP ViT-B-32 LAION-2B image embeddings',
                                      'dinov2_large_cosine': 'cosine similarity of DINOv2 ViT-L/14 global embeddings'},
               'models': {'DISTS': 'DISTS-pytorch default pretrained', 'DINOv2': 'facebookresearch/dinov2 dinov2_vitb14 pretrained',
                          'OpenCLIP': 'ViT-B-32 laion2b_s34b_b79k',
                          'DINOv2-Large': 'facebookresearch/dinov2 dinov2_vitl14 pretrained'},
               'limitations': 'Perceptual metrics use a cropped ROI, whereas SSIM/PSNR use a masked full frame; do not assume identical ROI or normalization.'}
    summary_path = run_folder / 'perceptual_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    result = {}
    for metric in engine.selected:
        col = METRIC_COLUMNS[metric]
        result['mean_' + col + '_all'] = average(rows, col)
        result['mean_' + col + '_fault_active'] = average(rows, col, active_only=True)
    return result
