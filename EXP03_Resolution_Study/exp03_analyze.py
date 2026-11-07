"""EXP03: reproducible offline paired image-scale and observed WebRTC resolution analysis.

Reads the existing experiment; never modifies capture images, original CSVs or status.
The main paired analysis compares the EXACT SAME reference/received frame pair
at multiple evaluation resolutions. It does NOT force WebRTC receiver resolution.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SCALES = [(320, 180), (640, 360), (960, 540), (1280, 720)]
METRICS = ['ssim', 'psnr_db', 'mae_rgb', 'dists_similarity', 'dinov2_cosine', 'dinov2_large_cosine', 'openclip_cosine']
PERCEPTUAL = {'DISTS': 'dists_similarity', 'DINOv2': 'dinov2_cosine',
              'DINOv2-Large': 'dinov2_large_cosine', 'OpenCLIP': 'openclip_cosine'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def import_py(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def mean(x):
    values = [v for v in (finite(a) for a in x) if v is not None]
    return statistics.mean(values) if values else None


def sd(x):
    values = [v for v in (finite(a) for a in x) if v is not None]
    return statistics.stdev(values) if len(values) >= 2 else None


def fmt(x, digits=4):
    v = finite(x)
    return f'{v:.{digits}f}' if v is not None else 'NA'


def mtable(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def resolve_run(root, name):
    path = Path(name)
    return path if path.is_absolute() else (root / path).resolve()


def plot_summary(output, run_rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[NOTE] matplotlib unavailable; PNG chart skipped.')
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for scenario, marker in [('baseline', 'o'), ('blur', 's')]:
        by_scale = defaultdict(list)
        for row in run_rows:
            if row['scenario'] == scenario:
                by_scale[row['target_width']].append(row['mean_ssim'])
        if not by_scale:
            continue
        xs = sorted(by_scale)
        ys = [mean(by_scale[x]) for x in xs]
        errs = [sd(by_scale[x]) or 0 for x in xs]
        ax.errorbar(xs, ys, yerr=errs, marker=marker, capsize=3,
                    label=f'{scenario} (run-level mean ± SD)')
    ax.set_xlabel('Evaluation width (pixels); downsample only')
    ax.set_ylabel('Masked SSIM')
    ax.set_xticks([w for w, _ in SCALES])
    ax.set_title('EXP03: Evaluation-scale sensitivity (same frame pairs)')
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'EXP03_scale_sensitivity.png', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Analyze EXP03 receiver-resolution and paired evaluation-scale effects')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--job', default='job_exp03_resolution_pilot_01', help='Name of dashboard_jobs subdirectory')
    ap.add_argument('--perceptual', nargs='*', choices=list(PERCEPTUAL), default=[],
                    help='Optional model metrics (slow on CPU): DISTS DINOv2 DINOv2-Large OpenCLIP')
    args = ap.parse_args()
    root = args.root.resolve()
    job = root / 'captures' / 'dashboard_jobs' / args.job
    status_path = job / 'status.json'
    if not status_path.is_file() and (job / 'status_recovery.json').is_file():
        status_path = job / 'status_recovery.json'
    if not status_path.is_file():
        ap.error('Job status missing; run the experiment first: ' + str(job))
    status = read_json(status_path)
    if status.get('state') != 'completed':
        ap.error(f'Job not completed (state={status.get("state")!r}); wait for completion before analysis')
    config = read_json(job / 'scenario.json')
    results = status.get('results') or []
    if not results:
        ap.error('No completed runs in status.json')
    quality = import_py(root / '08_jitsi_synchronized_quality.py', 'exp03_quality08')
    engine = None
    if args.perceptual:
        module = import_py(root / '12_perceptual_metrics.py', 'exp03_perceptual12')
        engine = module.PerceptualEngine(args.perceptual)
    source = Path(status.get('prepared_source') or (root / config['video']))
    if not source.is_absolute():
        source = (root / source).resolve()
    numbered = Path(status.get('numbered_source') or source.with_name(source.stem + '_numbered.y4m'))
    if not numbered.is_absolute():
        numbered = (root / numbered).resolve()
    metadata_path = numbered.with_name(numbered.stem + '_metadata.json')
    for p in (source, metadata_path):
        if not p.is_file():
            ap.error(f'Missing source or metadata: {p}')
    metadata = read_json(metadata_path)
    w, h, fps, offsets = quality.index_y4m(source)
    if (w, h) != (int(metadata['width']), int(metadata['height'])):
        ap.error('Source and barcode metadata dimensions differ')
    if len(offsets) != int(metadata['frame_count']):
        ap.error('Source and metadata frame counts differ')
    out = job / 'exp03_analysis'
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    native_frames = []
    diagnostics = []
    skipped = Counter()
    for run_i, result in enumerate(results, start=1):
        run_folder = resolve_run(root, result['run_folder'])
        if not run_folder.is_dir():
            skipped['missing_run_folder'] += 1
            print('[WARN] Missing:', run_folder)
            continue
        scenario = result.get('scenario')
        severity = result.get('severity')
        repeat = result.get('repeat_index', run_i)
        run_valid = result.get('run_valid', True)
        csv_path = run_folder / 'fault_measurements.csv'
        if not csv_path.is_file():
            skipped['missing_fault_measurements'] += 1
            print('[WARN] Missing:', csv_path)
            continue
        rows = read_csv(csv_path)
        resolutions = Counter()
        quality_ok = 0
        with source.open('rb') as f:
            for row in rows:
                if row.get('quality_status') != 'OK' or row.get('capture_status') != 'OK':
                    skipped['quality_not_ok'] += 1
                    continue
                if not run_valid:
                    skipped['invalid_run_samples'] += 1
                    continue
                idx = int(row['frame_index'])
                if not 0 <= idx < len(offsets):
                    skipped['frame_index_out_of_range'] += 1
                    continue
                png_path = run_folder / row['capture_filename']
                received = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
                if received is None:
                    skipped['png_unreadable'] += 1
                    continue
                native_h, native_w = received.shape[:2]
                if (native_w, native_h) != (int(row['capture_width']), int(row['capture_height'])):
                    skipped['dimension_mismatch'] += 1
                    continue
                resolutions[(native_w, native_h)] += 1
                quality_ok += 1
                reference = quality.read_reference(f, offsets[idx], w, h, native_w, native_h)
                native_mask, _ = quality.valid_pixel_mask(received.shape, metadata)
                native_ssim, native_psnr = quality.quality_metrics(reference, received, native_mask)
                logged_ssim = finite(row.get('ssim_gray_masked'))
                if logged_ssim is not None and abs(native_ssim - logged_ssim) > 1e-4:
                    skipped['logged_ssim_mismatch'] += 1
                native_frames.append({
                    'run_index': run_i, 'scenario': scenario,
                    'native_width': native_w, 'native_height': native_h,
                    'ssim': native_ssim, 'psnr_db': native_psnr,
                    'bitrate_mbps': row.get('bitrate_mbps', '')
                })
                # The same two arrays are downsampled together. No upscaling is used.
                for target_w, target_h in SCALES:
                    if target_w > native_w or target_h > native_h:
                        continue
                    if (target_w, target_h) == (native_w, native_h):
                        rec_small, ref_small = received, reference
                        mask = native_mask
                        ssim, psnr = native_ssim, native_psnr
                    else:
                        rec_small = cv2.resize(received, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        ref_small = cv2.resize(reference, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        mask, _ = quality.valid_pixel_mask(rec_small.shape, metadata)
                        ssim, psnr = quality.quality_metrics(ref_small, rec_small, mask)
                    mae = float(np.mean(np.abs(ref_small.astype(np.float32) - rec_small.astype(np.float32))[mask]))
                    rec = {
                        'run_index': run_i, 'run_folder': run_folder.name,
                        'scenario': scenario, 'severity': severity, 'repeat_index': repeat,
                        'sample': row['sample'], 'frame_index': idx,
                        'elapsed_s': row.get('elapsed_s', ''),
                        'native_width': native_w, 'native_height': native_h,
                        'target_width': target_w, 'target_height': target_h,
                        'fault_active': row.get('fault_active', ''),
                        'ssim': ssim, 'psnr_db': psnr if math.isfinite(psnr) else '',
                        'mae_rgb': mae, 'native_ssim': native_ssim,
                        'ssim_delta_vs_native': ssim - native_ssim,
                        'bitrate_mbps': row.get('bitrate_mbps', ''),
                        'observed_decode_fps': row.get('observed_decode_fps', ''),
                        'jitter_ms': row.get('jitter_ms', ''),
                        'interval_packet_loss_pct': row.get('interval_packet_loss_pct', ''),
                        'dists_similarity': '', 'dinov2_cosine': '',
                        'dinov2_large_cosine': '', 'openclip_cosine': '',
                    }
                    if engine is not None:
                        overlay_bottom = int(metadata['overlay_rect_xyxy_source_pixels'][3] * target_h / h)
                        rec.update(engine.compare(ref_small, rec_small, overlay_bottom))
                    frames.append(rec)
        diagnostics.append({
            'run_index': run_i, 'run_folder': run_folder.name, 'scenario': scenario,
            'severity': severity, 'repeat_index': repeat, 'run_valid': run_valid,
            'quality_ok_samples': quality_ok,
            'observed_resolutions': '; '.join(f'{a}x{b}:{count}' for (a,b), count in sorted(resolutions.items())),
            'resolution_switches': sum(1 for i in range(1, len(rows))
                if rows[i].get('capture_width') and rows[i-1].get('capture_width') and
                (rows[i].get('capture_width'), rows[i].get('capture_height')) !=
                (rows[i-1].get('capture_width'), rows[i-1].get('capture_height'))),
            'mean_logged_ssim': mean(r.get('ssim_gray_masked') for r in rows if r.get('quality_status') == 'OK'),
            'mean_bitrate_mbps': mean(r.get('bitrate_mbps') for r in rows),
            'mean_decode_fps': mean(r.get('observed_decode_fps') for r in rows),
        })
        print(f'[RUN {run_i:02d}/{len(results):02d}] {scenario} repeat={repeat}, '
              f'{quality_ok} usable frames, resolutions={dict(resolutions)}', flush=True)
    if not frames:
        ap.error('No usable frames; see warnings and capture directories')
    frame_fields = list(frames[0])
    write_csv(out / 'EXP03_frame_metrics.csv', frames, frame_fields)
    write_csv(out / 'EXP03_run_diagnostics.csv', diagnostics, list(diagnostics[0]))
    by_run_scale = defaultdict(list)
    by_native = defaultdict(list)
    for row in frames:
        by_run_scale[(row['run_index'], row['target_width'], row['target_height'])].append(row)
    for row in native_frames:
        by_native[(row['run_index'], row['native_width'], row['native_height'])].append(row)
    run_rows = []
    for (i, tw, th), group in sorted(by_run_scale.items()):
        rec = {'run_index': i, 'run_folder': group[0]['run_folder'],
               'scenario': group[0]['scenario'], 'repeat_index': group[0]['repeat_index'],
               'target_width': tw, 'target_height': th, 'n_frames': len(group),
               'mean_ssim': mean(r['ssim'] for r in group),
               'mean_psnr_db': mean(r['psnr_db'] for r in group),
               'mean_mae_rgb': mean(r['mae_rgb'] for r in group),
               'mean_ssim_delta_vs_native': mean(r['ssim_delta_vs_native'] for r in group),
               'native_resolutions_included': ';'.join(sorted({f'{r["native_width"]}x{r["native_height"]}' for r in group}))}
        for model_key in PERCEPTUAL.values():
            rec['mean_' + model_key] = mean(r[model_key] for r in group)
        run_rows.append(rec)
    write_csv(out / 'EXP03_run_level.csv', run_rows, list(run_rows[0]))
    native_rows = []
    for (i, nw, nh), group in sorted(by_native.items()):
        native_rows.append({'run_index': i, 'scenario': group[0]['scenario'],
                            'native_width': nw, 'native_height': nh, 'n_frames': len(group),
                            'mean_ssim': mean(r['ssim'] for r in group),
                            'mean_psnr_db': mean(r['psnr_db'] for r in group),
                            'mean_bitrate_mbps': mean(r['bitrate_mbps'] for r in group)})
    write_csv(out / 'EXP03_native_resolution_observed.csv', native_rows,
              ['run_index', 'scenario', 'native_width', 'native_height', 'n_frames', 'mean_ssim', 'mean_psnr_db', 'mean_bitrate_mbps'])
    # To avoid changing the set of frames across evaluation scales, compare
    # only frames captured at a common native resolution for each paired table.
    paired = []
    by_native_run = defaultdict(list)
    for row in frames:
        by_native_run[(row['run_index'], row['native_width'], row['native_height'],
                       row['target_width'], row['target_height'])].append(row)
    for (i, nw, nh, tw, th), group in sorted(by_native_run.items()):
        if (nw, nh) == (tw, th):
            continue
        paired.append({'run_index': i, 'scenario': group[0]['scenario'],
                       'native_width': nw, 'native_height': nh,
                       'target_width': tw, 'target_height': th,
                       'n_paired_frames': len(group),
                       'mean_ssim_at_native': mean(r['native_ssim'] for r in group),
                       'mean_ssim_at_target': mean(r['ssim'] for r in group),
                       'mean_paired_delta': mean(r['ssim_delta_vs_native'] for r in group)})
    write_csv(out / 'EXP03_paired_ssim_deltas.csv', paired,
              ['run_index', 'scenario', 'native_width', 'native_height', 'target_width', 'target_height',
               'n_paired_frames', 'mean_ssim_at_native', 'mean_ssim_at_target', 'mean_paired_delta'])
    # Report: run-level means; never use 160 correlated frames as independent n.
    summary = defaultdict(list)
    for r in run_rows:
        summary[(r['scenario'], r['target_width'], r['target_height'])].append(r)
    table = []
    for (scenario, tw, th), rs in sorted(summary.items()):
        table.append([scenario, f'{tw}×{th}', len(rs), sum(r['n_frames'] for r in rs),
                      fmt(mean(r['mean_ssim'] for r in rs)), fmt(sd(r['mean_ssim'] for r in rs)),
                      fmt(mean(r['mean_psnr_db'] for r in rs), 2)])
    obs_table = []
    for r in native_rows:
        obs_table.append([r['run_index'], r['scenario'],
                          f'{r["native_width"]}×{r["native_height"]}', r['n_frames'],
                          fmt(r['mean_ssim']), fmt(r['mean_bitrate_mbps'], 3)])
    delta_groups = defaultdict(list)
    for r in paired:
        delta_groups[(r['scenario'], r['native_width'], r['target_width'])].append(r)
    delta_table = []
    for (scenario, nw, tw), rs in sorted(delta_groups.items()):
        delta_table.append([scenario, f'{nw}→{tw}', len(rs),
                            sum(r['n_paired_frames'] for r in rs),
                            fmt(mean(r['mean_paired_delta'] for r in rs)),
                            fmt(sd(r['mean_paired_delta'] for r in rs))])
    warnings = []
    for scenario in ['baseline', 'blur']:
        count = sum(r['scenario'] == scenario and r['run_valid'] for r in diagnostics)
        if count < 5:
            warnings.append(f'{scenario}: only {count} valid sessions; pilot is underpowered for inferential claims.')
    if not any(r['native_width'] == 1280 for r in native_rows):
        warnings.append('No native 1280-pixel captures; 1280×720 controlled-scale comparison unavailable.')
    if skipped['logged_ssim_mismatch']:
        warnings.append(f'{skipped["logged_ssim_mismatch"]} frames had native SSIM disagreement > 1e-4; inspect mask/reference version.')
    if skipped['invalid_run_samples']:
        warnings.append(f'{skipped["invalid_run_samples"]} samples excluded because their Jitsi session was invalid.')
    warnings.append('Jitsi receiver resolution is observed, not randomized or forced; bitrate, network, and time can confound native-resolution differences.')
    warnings.append('The run order is baseline block then blur block; condition effects may be confounded by time/order.')
    warnings.append('Same-frame downsampling isolates the metric evaluation-scale intervention, NOT a WebRTC resolution intervention.')
    warnings.append('Paired scales must be compared within the same native-resolution cohort; scales have different eligible samples otherwise.')
    warnings.append('Independent experimental unit is the Jitsi session; frame observations within a session are correlated.')
    report = [
        '# EXP03 – Resolution Sensitivity Pilot: Analysis', '',
        f'Job: `{job.name}`',
        f'Valid/total Jitsi sessions: {sum(bool(r.get("run_valid", True)) for r in results)}/{len(results)}',
        f'Analyzed frame-scale rows: {len(frames)}',
        f'Original video: `{source.name}`',
        f'Perceptual models: {", ".join(args.perceptual) if args.perceptual else "not selected"}', '',
        '## 1. Evaluation-scale sensitivity (descriptive run-level summaries)', '',
        'The reference and received image of EACH sample were downsampled together; upscaling was disallowed. '
        'Mean and SD are computed across independent session means, not across correlated frames.', '',
        mtable(['Condition', 'Evaluation size', 'Sessions', 'Frame pairs', 'SSIM mean', 'SSIM SD', 'PSNR mean dB'], table), '',
        '## 2. Observed native receiver resolutions (non-causal)', '',
        mtable(['Session', 'Condition', 'Native size', 'Frames', 'SSIM mean', 'Bitrate mean Mbps'], obs_table), '',
        '## 3. Strictly paired SSIM deltas within a native-resolution cohort', '',
        'Delta = SSIM(after downsampling both images) - SSIM(at native capture resolution). '
        'Each row uses the same frame pairs; summaries aggregate per-session deltas.', '',
        mtable(['Condition', 'Native→target width', 'Sessions', 'Frame pairs', 'Mean delta', 'SD across sessions'], delta_table) if delta_table else 'No eligible native-to-smaller-size pairs.', '',
        '## 4. Interpretation and limitations', '',
        *[f'- {line}' for line in warnings], '',
        '## 5. Output files', '',
        '- `EXP03_frame_metrics.csv`: every valid frame at every eligible evaluation scale.',
        '- `EXP03_run_level.csv`: independent-session summaries by evaluation scale.',
        '- `EXP03_run_diagnostics.csv`: resolution switches, bitrate, validity.',
        '- `EXP03_native_resolution_observed.csv`: native receiver resolution strata.',
        '- `EXP03_paired_ssim_deltas.csv`: paired native-versus-downsampled deltas.',
        '- `EXP03_scale_sensitivity.png`: descriptive figure (if matplotlib installed).', '',
        '## 6. Next step', '',
        'If these results reveal large evaluation-scale effects, pre-register a fixed evaluation scale '
        'and perform an independent confirmatory Jitsi study. If a causal WebRTC-resolution '
        'effect is required, implement and VERIFY receiver-layer or sender-encoding controls '
        'and record actual negotiated inbound resolution; this pilot does not implement such controls.',
    ]
    (out / 'EXP03_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    plot_summary(out, run_rows)
    print('\n[DONE] EXP03 analysis:', out)
    print('[FILES] EXP03_report.md, EXP03_frame_metrics.csv, EXP03_run_level.csv,')
    print('        EXP03_run_diagnostics.csv, EXP03_native_resolution_observed.csv,')
    print('        EXP03_paired_ssim_deltas.csv, EXP03_scale_sensitivity.png (optional)')
    if skipped:
        print('[DIAGNOSTICS] Excluded/warning counts:', dict(skipped))
    print('[NOTE] Receiver resolution was NOT forced; read limitations in EXP03_report.md')


if __name__ == '__main__':
    main()
