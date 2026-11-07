#!/usr/bin/env python3
"""Read-only, reproducible offline latency benchmark for EXP04 visual metrics.

Run from the original Windows project (visual-oracle conda environment):
  python exp04_runtime_benchmark.py --check-only
  python exp04_runtime_benchmark.py --smoke
  python exp04_runtime_benchmark.py --include-dino-large

Does NOT start Jitsi, recalibrate thresholds, modify the plan/lock/scorer,
or overwrite any prior experimental artifact. A new ZIP is written under captures/.

Timing protocol:
 - 48 deterministically sampled, existing, eligible holdout frame pairs by default
   (4 meetings per video x 6 conditions; 1 frame per meeting).
 - Per-stage CUDA-synchronized wall time; models warmed up; inference_mode.
 - Separate disk decoding, preprocessing, SSIM, PSNR, DISTS, DINOv2-Base,
   OpenCLIP, optional DINOv2-Large, and a complete in-memory offline pipeline.
 - Both reference+receiver inference and cached-reference receiver-only latency.
 - Batch 1 and 2 throughput of the deep inference trio, excluding disk/preprocess.
 - All results are descriptive for THIS workstation, not full WebRTC pipeline
   latency and not a proof of continuous 25/30 FPS monitoring.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict, Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np

import exp04_deep_calibration as deep
import exp04_score_holdout as hold

RAW_COLS = ['stage','repeat','job','video','condition','severity','frame_index',
            'native_width','native_height','n_pairs','milliseconds','gpu_peak_allocated_mb']
SUMMARY_COLS = ['stage','observations','pairs_processed','mean_ms','median_ms',
                'p90_ms','p95_ms','min_ms','max_ms','throughput_pairs_per_second',
                'gpu_peak_allocated_mb']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def choose_frames(jobs, prepared, count, seed):
    if count%12 or not 12<=count<=120:
        raise ValueError('--pairs must be a multiple of 12 between 12 and 120')
    rng=random.Random(seed)
    by=defaultdict(list)
    for j in jobs:
        by[(j['video'],j['condition'])].append(j)
    selected=[]
    each=count//12
    for video in ('parkrun','screenshare'):
        for condition in ('baseline','black','blur','color','freeze','occlusion'):
            options=sorted(by[(video,condition)],key=lambda x:x['name'])
            rng.shuffle(options)
            if len(options)<each:
                raise RuntimeError(f'Insufficient jobs for {video}/{condition}: {len(options)}')
            for j in options[:each]:
                rows=prepared[j['name']]['rows']
                # For faults, prefer a frame known to be inside the injection window.
                candidates=[r for r in rows if int(r['fault_active'])==1] if condition!='baseline' else rows
                if not candidates:
                    candidates=rows
                r=candidates[len(candidates)//2]
                selected.append((j,prepared[j['name']],r))
    return selected


def ssim_only(ref, received, mask):
    # Exact SSIM branch of 08_jitsi_synchronized_quality.quality_metrics.
    x=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY).astype(np.float32)
    y=cv2.cvtColor(received,cv2.COLOR_BGR2GRAY).astype(np.float32)
    mx=cv2.GaussianBlur(x,(11,11),1.5)
    my=cv2.GaussianBlur(y,(11,11),1.5)
    sx=cv2.GaussianBlur(x*x,(11,11),1.5)-mx*mx
    sy=cv2.GaussianBlur(y*y,(11,11),1.5)-my*my
    sxy=cv2.GaussianBlur(x*y,(11,11),1.5)-mx*my
    c1,c2=(.01*255)**2,(.03*255)**2
    smap=((2*mx*my+c1)*(2*sxy+c2))/np.maximum(
        (mx**2+my**2+c1)*(sx+sy+c2),1e-9)
    return float(np.mean(smap[mask]))


def psnr_only(ref,received,mask):
    # Exact PSNR branch of 08_jitsi_synchronized_quality.quality_metrics.
    diff=ref.astype(np.float64)-received.astype(np.float64)
    mse=float(np.mean((diff[mask])**2))
    return math.inf if mse==0 else 10*math.log10(255**2/mse)


def gpu_sync(torch,device):
    if device=='cuda':torch.cuda.synchronize()


def timed(torch,device,fn):
    gpu_sync(torch,device)
    start=time.perf_counter()
    result=fn()
    gpu_sync(torch,device)
    return result,(time.perf_counter()-start)*1000.0


def get_gpu_peak(torch,device):
    return float(torch.cuda.max_memory_allocated()/1024**2) if device=='cuda' else None


def reset_gpu_peak(torch,device):
    if device=='cuda':
        gpu_sync(torch,device)
        torch.cuda.reset_peak_memory_stats()


def record(raw,stage,rep,pair,ms,peak,n=1):
    j=pair['job'] if pair else None
    r=pair['meta'] if pair else None
    raw.append({'stage':stage,'repeat':rep,'job':j['name'] if j else '',
                'video':j['video'] if j else '',
                'condition':j['condition'] if j else '',
                'severity':j['severity'] if j else '',
                'frame_index':int(r['frame_index']) if r else '',
                'native_width':int(float(r['capture_width'])) if r else '',
                'native_height':int(float(r['capture_height'])) if r else '',
                'n_pairs':n,'milliseconds':round(ms,6),
                'gpu_peak_allocated_mb':round(peak,3) if peak is not None else ''})


def load_pairs(chosen,sources,q,raw):
    pairs=[]
    with ExitStack() as stack:
        handles={v:stack.enter_context(sources[v]['path'].open('rb')) for v in sources}
        for j,item,r in chosen:
            def load():
                native=(int(float(r['capture_width'])),int(float(r['capture_height'])))
                im=cv2.imread(str(item['folder']/r['capture_filename']),cv2.IMREAD_COLOR)
                if im is None or im.shape[1::-1]!=native:
                    raise RuntimeError(f'PNG missing/changed: {j["name"]}/{r["sample"]}')
                if native!=hold.TARGET:
                    im=cv2.resize(im,hold.TARGET,interpolation=cv2.INTER_AREA)
                src=sources[j['video']]
                idx=int(r['frame_index'])
                ref=q.read_reference(handles[j['video']],src['offsets'][idx],
                                     src['w'],src['h'],*hold.TARGET)
                return ref,im
            (ref,im),ms=timed(None,'cpu',load)
            pair={'job':j,'meta':r,'ref':ref,'received':im}
            record(raw,'disk_png_y4m_decode_resize_once',0,pair,ms,None)
            pairs.append(pair)
    return pairs


def prepare_all(pairs,mask,clip_preprocess,torch,raw):
    for pair in pairs:
        pp,ms=timed(torch,'cpu',lambda:deep.prepare_pair(
            pair['ref'],pair['received'],mask,clip_preprocess,torch))
        pair['prepared']=pp
        record(raw,'preprocess_all_deep_models_once',0,pair,ms,None)


def dists_fn(pair,models,device,torch):
    dists=models[0]
    p=pair['prepared']
    with torch.inference_mode():
        a=p[0].unsqueeze(0).to(device)
        b=p[1].unsqueeze(0).to(device)
        out=dists(a,b,require_grad=False,batch_average=False)
        return float(out.reshape(-1)[0].item())


def dino_fn(pair,model,device,torch,F,ref_emb=None):
    p=pair['prepared']
    with torch.inference_mode():
        if ref_emb is None:
            x=torch.stack([p[2],p[3]]).to(device)
            v=F.normalize(model(x).float(),dim=-1)
            return float((v[0]*v[1]).sum().item())
        v=F.normalize(model(p[3].unsqueeze(0).to(device)).float(),dim=-1)
        return float((ref_emb*v[0]).sum().item())


def clip_fn(pair,model,device,torch,F,ref_emb=None):
    p=pair['prepared']
    with torch.inference_mode():
        if ref_emb is None:
            x=torch.stack([p[4],p[5]]).to(device)
            v=F.normalize(model.encode_image(x).float(),dim=-1)
            return float((v[0]*v[1]).sum().item())
        v=F.normalize(model.encode_image(p[5].unsqueeze(0).to(device)).float(),dim=-1)
        return float((ref_emb*v[0]).sum().item())


def reference_embeddings(pairs,dino,clip,device,torch,F,raw):
    for pair in pairs:
        p=pair['prepared']
        def calc_dino():
            with torch.inference_mode():
                return F.normalize(dino(p[2].unsqueeze(0).to(device)).float(),dim=-1)[0]
        def calc_clip():
            with torch.inference_mode():
                return F.normalize(clip.encode_image(p[4].unsqueeze(0).to(device)).float(),dim=-1)[0]
        emb,ms=timed(torch,device,calc_dino)
        pair['dino_ref']=emb
        record(raw,'dinov2_base_reference_embedding_once',0,pair,ms,get_gpu_peak(torch,device))
        emb,ms=timed(torch,device,calc_clip)
        pair['clip_ref']=emb
        record(raw,'openclip_reference_embedding_once',0,pair,ms,get_gpu_peak(torch,device))


def full_offline(pair,mask,models,device,torch,q):
    q.quality_metrics(pair['ref'],pair['received'],mask)
    p=deep.prepare_pair(pair['ref'],pair['received'],mask,models[3],torch)
    temporary={'prepared':p}
    dists_fn(temporary,models,device,torch)
    dino_fn(temporary,models[1],device,torch,models[5])
    clip_fn(temporary,models[2],device,torch,models[5])


def batch_three(chunk,models,device,torch):
    dists,dino,clip,_,_,F=models
    with torch.inference_mode():
        a=torch.stack([x['prepared'][0] for x in chunk]).to(device)
        b=torch.stack([x['prepared'][1] for x in chunk]).to(device)
        dists(a,b,require_grad=False,batch_average=False)
        din=torch.stack([p for x in chunk for p in (x['prepared'][2],x['prepared'][3])]).to(device)
        v=F.normalize(dino(din).float(),dim=-1)
        (v[0::2]*v[1::2]).sum(dim=-1)
        cli=torch.stack([p for x in chunk for p in (x['prepared'][4],x['prepared'][5])]).to(device)
        c=F.normalize(clip.encode_image(cli).float(),dim=-1)
        (c[0::2]*c[1::2]).sum(dim=-1)


def benchmark_large(pairs,args,device,torch,F,raw):
    print('[OPTIONAL] Loading DINOv2-Large (may download checkpoint)...',flush=True)
    start=time.perf_counter()
    model=torch.hub.load(deep.DINO_REPO,'dinov2_vitl14',trust_repo=True).eval().to(device)
    load_s=time.perf_counter()-start
    # The large model is for a separate, descriptive timing comparison only.
    # It was not part of the EXP04 frozen classifier or its model-weight lock.
    fp=deep.model_fingerprint(model)
    def run(pair):return dino_fn(pair,model,device,torch,F)
    for pair in pairs[:min(4,len(pairs))]:run(pair)
    for rep in range(args.repeats):
        for pair in pairs:
            _,ms=timed(torch,device,lambda p=pair:run(p))
            record(raw,'dinov2_large_reference_and_receiver',rep,pair,ms,get_gpu_peak(torch,device))
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return {'model':'dinov2_vitl14','weights_sha256':fp,'load_seconds':round(load_s,3)}


def write_csv(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        w.writeheader();w.writerows(rows)


def summary(raw):
    grouped=defaultdict(list)
    for r in raw:grouped[r['stage']].append(r)
    result=[]
    for stage,rows in sorted(grouped.items()):
        vals=np.array([float(x['milliseconds']) for x in rows],dtype=float)
        peak=[float(x['gpu_peak_allocated_mb']) for x in rows if x['gpu_peak_allocated_mb']!='']
        pairs=sum(int(x['n_pairs']) for x in rows)
        result.append({
            'stage':stage,'observations':len(rows),'pairs_processed':pairs,
            'mean_ms':round(float(vals.mean()),4),'median_ms':round(float(np.median(vals)),4),
            'p90_ms':round(float(np.percentile(vals,90)),4),
            'p95_ms':round(float(np.percentile(vals,95)),4),
            'min_ms':round(float(vals.min()),4),'max_ms':round(float(vals.max()),4),
            'throughput_pairs_per_second':round(float(1000*pairs/vals.sum()),4),
            'gpu_peak_allocated_mb':round(max(peak),2) if peak else ''})
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=r'D:\visual-oracle-revision')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='12 pairs x 1 repeat; no ZIP')
    ap.add_argument('--pairs',type=int,default=48,help='12..120 in steps of 12; default 48')
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--warmup',type=int,default=4)
    ap.add_argument('--batches',default='1,2',help='Comma-separated batch sizes 1,2,4; default 1,2')
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--include-dino-large',action='store_true',help='Optional DINOv2-Large timing; separate from EXP04 lock')
    args=ap.parse_args()
    if args.repeats<1 or args.repeats>10 or args.warmup<1 or args.warmup>16:
        ap.error('--repeats must be 1..10; --warmup must be 1..16')
    try:batches=[int(s.strip()) for s in args.batches.split(',')]
    except ValueError:ap.error('Invalid --batches; use e.g. 1,2')
    if not batches or any(x not in (1,2,4) for x in batches):
        ap.error('--batches may only contain 1,2,4')
    if args.smoke:
        args.pairs=12;args.repeats=1;args.warmup=2;args.include_dino_large=False
    root=Path(args.root).resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    os.chdir(root);sys.path.insert(0,str(root))
    print('[LOCK] Checking original plan, lock, frozen scorer and source files...',flush=True)
    plan,lock=hold.check_lock(root)
    q=deep.import_quality(root)
    sys.modules['exp04_quality_offline']=q
    jobs,sources,mask,mask_hash,prepared,exclusions=hold.prepare_holdout(root,plan,lock,q)
    total=sum(len(x['rows']) for x in prepared.values())
    if total!=3538:raise RuntimeError(f'Expected 3538 eligible holdout frames, got {total}')
    chosen=choose_frames(jobs,prepared,args.pairs,seed=20261107)
    print(f'[PREFLIGHT] PASS: {len(jobs)} COMPLETE meetings; {total} eligible frames; '
          f'{len(chosen)} sampled sessions; mask {mask.mean():.6f}; '
          f'excluded={dict(exclusions)}',flush=True)
    if args.check_only:
        print('[CHECK-ONLY] No models loaded and no files written.',flush=True)
        return

    import torch
    import torch.nn.functional as F
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else args.device
    if device=='auto':device='cpu'
    if device=='cpu':print('[WARNING] CPU timing may be very slow.',flush=True)
    torch.manual_seed(20261107)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(20261107)
    torch.backends.cudnn.benchmark=False
    raw=[]
    print('[INPUT] Reading existing PNG/Y4M pairs, no network sessions...',flush=True)
    pairs=load_pairs(chosen,sources,q,raw)
    for pair in pairs:
        s,p=q.quality_metrics(pair['ref'],pair['received'],mask)
        if abs(s-ssim_only(pair['ref'],pair['received'],mask))>1e-9:
            raise RuntimeError('SSIM split implementation differs from original')
        if abs(p-psnr_only(pair['ref'],pair['received'],mask))>1e-9:
            raise RuntimeError('PSNR split implementation differs from original')
    print('[MODELS] Loading and verifying locked DISTS, DINOv2-Base, OpenCLIP...',flush=True)
    start=time.perf_counter()
    models=deep.load_models(device)
    model_load_s=time.perf_counter()-start
    for k in ('dists_weights_sha256','dinov2_base_weights_sha256','openclip_weights_sha256'):
        if models[4][k]!=lock['calibration_evidence'][k]:
            raise RuntimeError(f'Model fingerprint differs from frozen calibration: {k}')
    prepare_all(pairs,mask,models[3],torch,raw)
    dino,clip=models[1],models[2]
    # Ensure GPU kernels and caches have warmed up before recorded inference.
    print('[WARMUP] Synchronizing CUDA and warming up kernels...',flush=True)
    for pair in pairs[:min(args.warmup,len(pairs))]:
        ssim_only(pair['ref'],pair['received'],mask)
        psnr_only(pair['ref'],pair['received'],mask)
        dists_fn(pair,models,device,torch)
        dino_fn(pair,dino,device,torch,F)
        clip_fn(pair,clip,device,torch,F)
    gpu_sync(torch,device)
    reference_embeddings(pairs,dino,clip,device,torch,F,raw)

    stages=[
        ('ssim_cpu',lambda p:ssim_only(p['ref'],p['received'],mask)),
        ('psnr_cpu',lambda p:psnr_only(p['ref'],p['received'],mask)),
        ('dists_reference_and_receiver',lambda p:dists_fn(p,models,device,torch)),
        ('dinov2_base_reference_and_receiver',lambda p:dino_fn(p,dino,device,torch,F)),
        ('openclip_reference_and_receiver',lambda p:clip_fn(p,clip,device,torch,F)),
        ('dinov2_base_receiver_cached_reference',lambda p:dino_fn(p,dino,device,torch,F,p['dino_ref'])),
        ('openclip_receiver_cached_reference',lambda p:clip_fn(p,clip,device,torch,F,p['clip_ref'])),
        ('full_offline_compute_no_disk',lambda p:full_offline(p,mask,models,device,torch,q)),
    ]
    print('[BENCHMARK] Isolated per-metric latency and complete offline computation...',flush=True)
    for stage,fn in stages:
        reset_gpu_peak(torch,device)
        for rep in range(args.repeats):
            for pair in pairs:
                _,ms=timed(torch,device,lambda p=pair:fn(p))
                record(raw,stage,rep,pair,ms,get_gpu_peak(torch,device))
        vals=[r['milliseconds'] for r in raw if r['stage']==stage]
        print(f'  {stage}: median {np.median(vals):.2f} ms; p95 {np.percentile(vals,95):.2f} ms',flush=True)

    print('[WORKLOAD] Deep trio inference only, varying batch size...',flush=True)
    skipped_batches=[]
    for batch_size in batches:
        try:
            # One warmup batch per configuration, not included in the measurements.
            batch_three(pairs[:batch_size],models,device,torch)
            reset_gpu_peak(torch,device)
            for rep in range(args.repeats):
                for start in range(0,len(pairs),batch_size):
                    chunk=pairs[start:start+batch_size]
                    _,ms=timed(torch,device,lambda c=chunk:batch_three(c,models,device,torch))
                    record(raw,f'deep_trio_batch_{batch_size}_no_disk_preprocess',rep,
                           chunk[0],ms,get_gpu_peak(torch,device),n=len(chunk))
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                skipped_batches.append({'batch':batch_size,'reason':'CUDA OOM'})
                print(f'[WARNING] Batch {batch_size} skipped: CUDA out of memory',flush=True)
                if device=='cuda':torch.cuda.empty_cache()
            else:raise

    large_info=None
    if args.include_dino_large:
        try:
            reset_gpu_peak(torch,device)
            large_info=benchmark_large(pairs,args,device,torch,F,raw)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                large_info={'status':'NOT_MEASURED_CUDA_OOM','error':str(e)[:300]}
                print('[WARNING] DINOv2-Large could not be measured due to CUDA OOM',flush=True)
                if device=='cuda':torch.cuda.empty_cache()
            else:raise
    report=summary(raw)
    if args.smoke:
        print('[SMOKE] PASS: completed all timing stages; no files written.',flush=True)
        for x in report:print(f"  {x['stage']}: p50={x['median_ms']}ms, p95={x['p95_ms']}ms")
        return

    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir=root/'captures'/f'exp04_runtime_benchmark_{stamp}'
    outdir.mkdir(parents=True,exist_ok=False)
    raw_path=outdir/'EXP04_runtime_raw.csv'
    summary_path=outdir/'EXP04_runtime_summary.csv'
    audit_path=outdir/'EXP04_runtime_audit.json'
    readme_path=outdir/'EXP04_runtime_README.txt'
    checksums_path=outdir/'EXP04_runtime_checksums.json'
    write_csv(raw_path,RAW_COLS,raw)
    write_csv(summary_path,SUMMARY_COLS,report)
    hardware={'device':device,'cuda_available':torch.cuda.is_available(),
              'gpu_name':torch.cuda.get_device_name(0) if device=='cuda' else None,
              'gpu_total_memory_gb':round(torch.cuda.get_device_properties(0).total_memory/1024**3,3) if device=='cuda' else None,
              'gpu_max_memory_allocated_mb':round(torch.cuda.max_memory_allocated()/1024**2,2) if device=='cuda' else None}
    try:
        import psutil
        hardware['process_rss_mb_at_end']=round(psutil.Process().memory_info().rss/1024**2,2)
        hardware['cpu_logical_count']=psutil.cpu_count(logical=True)
    except ImportError:hardware['process_rss_mb_at_end']=None
    audit={
        'protocol':'EXP04_readonly_offline_runtime_benchmark_v1',
        'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'plan_sha256':hold.PLAN_SHA256,'lock_sha256':hold.LOCK_SHA256,
        'frozen_scorer_sha256':hold.FROZEN_SHA256,
        'mask_sha256':mask_hash,'locked_model_weight_fingerprints':models[4],
        'original_complete_holdout_sessions':120,'original_eligible_frames':total,
        'sampled_independent_sessions':len(pairs),
        'sample_selection_seed':20261107,
        'sampled_by_video_condition':{f'{v}/{c}':sum(p['job']['video']==v and p['job']['condition']==c for p in pairs)
                                      for v in ('parkrun','screenshare')
                                      for c in ('baseline','black','blur','color','freeze','occlusion')},
        'repeats':args.repeats,'warmup_pairs':args.warmup,'batch_sizes':batches,
        'skipped_batch_sizes':skipped_batches,
        'model_loading_and_fingerprinting_seconds':round(model_load_s,3),
        'optional_dinov2_large':large_info,
        'hardware':hardware,'python':sys.version,'platform':platform.platform(),
        'torch':torch.__version__,'opencv':cv2.__version__,
        'note':('CUDA-synchronized wall clock, offline, sequential execution; model loading excluded from '
                'per-frame latencies. Deep trio batch throughput excludes disk, image capture, and '
                'preprocessing. No claim of live full-frame-rate WebRTC throughput.'),
        'experiment_data_modified':False,'new_jitsi_sessions_started':False,
    }
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readme_path.write_text(
        'EXP04 READ-ONLY OFFLINE PERFORMANCE BENCHMARK\n'
        '=========================================\n'
        'SSIM/PSNR: exact original formulas, separated for CPU timing.\n'
        'DISTS, DINOv2-Base and OpenCLIP: original locked models and preprocessing.\n'
        'Reference+receiver: two image embeddings per pair, where applicable.\n'
        'Receiver_cached_reference: reference embeddings precomputed; reference\n'
        'precomputation latency is separately reported.\n'
        'full_offline_compute_no_disk: includes SSIM+PSNR, preprocessing and all\n'
        'three deep metric computations; excludes image read and live capture.\n'
        'deep_trio_batch_B: three deep models only, B pairs per batch; excludes\n'
        'SSIM/PSNR, disk, preprocessing and network/capture overhead.\n'
        'GPU times include synchronization and CPU-to-GPU tensor transfer.\n'
        'Batch results are descriptive and cannot establish sustained live FPS.\n'
        'Optional DINOv2-Large is NOT part of EXP04 frozen oracle.\n'
        'Do not add these post-hoc runtime results to the frozen primary accuracy\n'
        'evaluation or modify thresholds.\n',encoding='utf-8')
    files=[raw_path,summary_path,audit_path,readme_path]
    checksums_path.write_text(json.dumps({p.name:sha256(p) for p in files},indent=2)+'\n',encoding='utf-8')
    zpath=root/'captures'/f'EXP04_runtime_benchmark_{stamp}.zip'
    with ZipFile(zpath,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[checksums_path]:z.write(p,p.name)
    print(f'[COMPLETE] {len(pairs)} existing sessions; {len(raw)} timing records; '
          f'new Jitsi sessions=0.\nUPLOAD: {zpath}',flush=True)

if __name__=='__main__':main()
