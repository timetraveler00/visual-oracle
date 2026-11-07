#!/usr/bin/env python3
"""EXP04 Shields: independent receiver-side temporal validation.

Run from Windows Anaconda Prompt (with Jitsi and Selenium Grid already running):
    cd /d D:\\visual-oracle-revision
    python exp04_temporal_validation.py

Runs two separate Jitsi sessions (baseline and freeze/low), each 35 seconds
of receiver-side requestVideoFrameCallback observations. Does NOT edit any
existing experiment script, Y4M, scenario, or prior capture directory.

Measures *source-content freeze*, not a WebRTC transport freeze. The frame
counter is excluded from scene MAE but decoded independently to prove that
source frame IDs continue advancing while scene content is held.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_FIELDS = [
    'condition', 'seq', 'relative_ms', 'presentation_ms', 'media_time_s',
    'presented_frames', 'callback_gap_ms', 'frame_index', 'barcode_confidence',
    'barcode_error', 'scene_mae', 'video_width', 'video_height', 'track_id',
    'scene_valid_pixels', 'callback_processing_ms', 'callback_error',
]

START_JS = r"""
const opts = arguments[0];
if (window.__exp04Temporal && window.__exp04Temporal.running) {
  return {ok:false, error:'A temporal observer is already running'};
}
const candidates = [...document.querySelectorAll('video')].map(v => {
  const parent = v.closest('.videocontainer, .remote-video-container');
  const id = v.id || '', cls = String(v.className || '');
  const parentId = parent ? (parent.id || '') : '';
  const parentCls = parent ? String(parent.className || '') : '';
  const remote = /remote/i.test(id) || /remote/i.test(cls) ||
                 /remote/i.test(parentId) || /remote/i.test(parentCls) ||
                 /^participant_/.test(parentId);
  const local = /local/i.test(id) || /local/i.test(cls) ||
                /local/i.test(parentId) || /local/i.test(parentCls);
  return {v, remote, local, id, parentId};
}).filter(x => x.remote && !x.local && x.v.readyState >= 2 &&
                   x.v.videoWidth > 0 && x.v.videoHeight > 0);
candidates.sort((a,b) => b.v.videoWidth*b.v.videoHeight - a.v.videoWidth*a.v.videoHeight);
if (!candidates.length) return {ok:false, error:'No ready REMOTE video element'};
const chosen = candidates[0];
const v = chosen.v;
if (typeof v.requestVideoFrameCallback !== 'function') {
  return {ok:false, error:'Chrome requestVideoFrameCallback is not supported'};
}
const trackIds = (v.srcObject && typeof v.srcObject.getVideoTracks === 'function') ?
  v.srcObject.getVideoTracks().map(t => t.id) : [];
const cfg = opts.metadata;
const srcW = Number(cfg.width), srcH = Number(cfg.height);
const bc = cfg.barcode;
if (!bc || Number(bc.number_of_cells) !== 24) {
  return {ok:false, error:'24-cell frame barcode metadata missing'};
}
const sceneW = 320, sceneH = 180, barcodeW = 640, barcodeH = 360;
const sc = document.createElement('canvas'); sc.width=sceneW; sc.height=sceneH;
const sx = sc.getContext('2d', {willReadFrequently:true});
const cb = document.createElement('canvas'); cb.width=barcodeW; cb.height=barcodeH;
const cx = cb.getContext('2d', {willReadFrequently:true});
const r = cfg.overlay_rect_xyxy_source_pixels;
if (!r || r.length !== 4) return {ok:false,error:'overlay_rect_xyxy_source_pixels missing'};
const xa = Math.max(0, Math.floor(r[0]*sceneW/srcW)-2);
const ya = Math.max(0, Math.floor(r[1]*sceneH/srcH)-2);
const xb = Math.min(sceneW, Math.ceil(r[2]*sceneW/srcW)+2);
const yb = Math.min(sceneH, Math.ceil(r[3]*sceneH/srcH)+2);
const mask = new Uint8Array(sceneW*sceneH);
let validPixels=0;
for (let y=0; y<sceneH; y++) for (let x=0; x<sceneW; x++) {
  const valid = !(x>=xa && x<xb && y>=ya && y<yb);
  mask[y*sceneW+x]=valid ? 1:0;
  if (valid) validPixels++;
}
if (validPixels < 0.5*sceneW*sceneH) return {ok:false,error:'Overlay mask covers too much image'};
function decodeBarcode() {
  const values=[];
  for (let i=0; i<24; i++) {
    const x0 = Math.max(0, Math.min(barcodeW-1, Math.round((Number(bc.source_cell_x)+(i+0.325)*Number(bc.source_cell_width))*barcodeW/srcW)));
    const x1 = Math.max(x0+1, Math.min(barcodeW, Math.round((Number(bc.source_cell_x)+(i+0.675)*Number(bc.source_cell_width))*barcodeW/srcW)));
    const y0 = Math.max(0, Math.min(barcodeH-1, Math.round((Number(bc.source_cell_y)+0.325*Number(bc.source_cell_height))*barcodeH/srcH)));
    const y1 = Math.max(y0+1, Math.min(barcodeH, Math.round((Number(bc.source_cell_y)+0.675*Number(bc.source_cell_height))*barcodeH/srcH)));
    const img = cx.getImageData(x0,y0,x1-x0,y1-y0).data;
    const samples=[];
    for (let j=0; j<img.length; j+=4) samples.push((77*img[j]+150*img[j+1]+29*img[j+2])/256);
    samples.sort((a,b)=>a-b);
    values.push(samples[Math.floor(samples.length/2)]);
  }
  const low=Math.min(...values), high=Math.max(...values);
  if (high-low<60) return {index:null,confidence:null,error:'low barcode contrast '+(high-low).toFixed(1)};
  const threshold=(high+low)/2;
  const bits=values.map(x=>x>threshold?'1':'0').join('');
  if (bits.slice(0,4)!=='1010') return {index:null,confidence:null,error:'invalid guard '+bits.slice(0,4)};
  const index=parseInt(bits.slice(4,20),2), check=parseInt(bits.slice(20,24),2);
  const expected=((index>>12)^(index>>8)^(index>>4)^index)&15;
  if (check!==expected) return {index:null,confidence:null,error:'barcode checksum mismatch'};
  if (index<0 || index>=Number(cfg.frame_count)) return {index:null,confidence:null,error:'frame index out of range '+index};
  return {index,confidence:Math.min(...values.map(x=>Math.abs(x-threshold))),error:''};
}
const state={running:true,rows:[],count:0,overflow:0,error:'',callbackId:null,
  startedPerf:performance.now(),lastNow:null,lastGray:null,video:v,
  videoId:chosen.id,parentId:chosen.parentId,trackIds,validPixels,
  sceneSize:[sceneW,sceneH],barcodeSize:[barcodeW,barcodeH],maskRect:[xa,ya,xb,yb]};
window.__exp04Temporal=state;
function tick(now, md) {
  if (!state.running) return;
  const row={seq:++state.count,relative_ms:now-state.startedPerf,
    presentation_ms:typeof md.presentationTime==='number'?md.presentationTime:null,
    media_time_s:typeof md.mediaTime==='number'?md.mediaTime:null,
    presented_frames:typeof md.presentedFrames==='number'?md.presentedFrames:null,
    callback_gap_ms:state.lastNow===null?null:now-state.lastNow,
    frame_index:null,barcode_confidence:null,barcode_error:'',scene_mae:null,
    video_width:v.videoWidth,video_height:v.videoHeight,
    track_id:trackIds.length?trackIds[0]:'',scene_valid_pixels:validPixels,callback_error:''};
  state.lastNow=now;
  const processStart=performance.now();
  try {
    if (!v.isConnected || v.readyState<2) throw new Error('Remote video disconnected or not ready');
    cx.drawImage(v,0,0,barcodeW,barcodeH);
    const b=decodeBarcode();
    row.frame_index=b.index;
    row.barcode_confidence=b.confidence;
    row.barcode_error=b.error;
    sx.drawImage(v,0,0,sceneW,sceneH);
    const pixels=sx.getImageData(0,0,sceneW,sceneH).data;
    const gray=new Uint8Array(sceneW*sceneH);
    let sum=0;
    for (let i=0,j=0; i<gray.length; i++,j+=4) {
      const g=(77*pixels[j]+150*pixels[j+1]+29*pixels[j+2]+128)>>8;
      gray[i]=g;
      if (state.lastGray!==null && mask[i]) sum+=Math.abs(g-state.lastGray[i]);
    }
    if (state.lastGray!==null) row.scene_mae=sum/validPixels;
    state.lastGray=gray;
  } catch(e) {
    row.callback_error=String(e);
    state.error=String(e);
    state.running=false;
  }
  row.callback_processing_ms=performance.now()-processStart;
  if (state.rows.length<6000) state.rows.push(row); else state.overflow++;
  if (state.running) state.callbackId=v.requestVideoFrameCallback(tick);
}
state.callbackId=v.requestVideoFrameCallback(tick);
return {ok:true,videoId:state.videoId,parentId:state.parentId,trackIds,
  videoWidth:v.videoWidth,videoHeight:v.videoHeight,maskRect:state.maskRect,
  sceneValidPixels:validPixels,remoteCandidates:candidates.length};
"""

DRAIN_JS = r"""
const s=window.__exp04Temporal;
if (!s) return {ok:false,error:'Temporal observer not initialized'};
return {ok:true,running:s.running,count:s.count,overflow:s.overflow,error:s.error,
  connected:!!s.video.isConnected,lastCallbackAgeMs:s.lastNow===null?null:performance.now()-s.lastNow,
  rows:s.rows.splice(0,s.rows.length)};
"""
STOP_JS = r"""
const s=window.__exp04Temporal;
if (!s) return {ok:false,error:'Temporal observer not initialized'};
s.running=false;
if (s.callbackId!==null && typeof s.video.cancelVideoFrameCallback==='function') {
  try {s.video.cancelVideoFrameCallback(s.callbackId);}catch(e){}
}
return {ok:true,running:s.running,count:s.count,overflow:s.overflow,error:s.error,
  connected:!!s.video.isConnected,lastCallbackAgeMs:s.lastNow===null?null:performance.now()-s.lastNow,
  rows:s.rows.splice(0,s.rows.length)};
"""


def load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_save(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def median(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return round(statistics.median(vals), 6) if vals else None


def pct(values, p):
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    t = (len(vals)-1)*p/100
    lo, hi = int(math.floor(t)), int(math.ceil(t))
    return round(vals[lo]*(hi-t)+vals[hi]*(t-lo), 6) if lo!=hi else round(vals[lo], 6)


def resolve_inputs(args):
    numbered = Path(args.numbered).resolve() if args.numbered else ROOT/'videos'/'720p50_shields_ter_numbered.y4m'
    metadata_path = Path(args.metadata).resolve() if args.metadata else numbered.with_name(numbered.stem+'_metadata.json')
    if args.fault_video:
        fault_video = Path(args.fault_video).resolve()
    else:
        candidates = sorted((ROOT/'fault_sources').glob('**/720p50_shields_ter_fault_freeze_low.y4m'))
        if len(candidates) != 1:
            raise RuntimeError('Expected exactly one Shields freeze/low Y4M under fault_sources; found '
                               f'{len(candidates)}. Specify --fault-video with its full path. '
                               f'Candidates: {[str(p) for p in candidates]}')
        fault_video = candidates[0]
    for p in (numbered, metadata_path, fault_video):
        if not p.is_file():
            raise FileNotFoundError(f'Required file not found: {p}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    for key in ('width','height','fps','frame_count','barcode','overlay_rect_xyxy_source_pixels'):
        if key not in metadata:
            raise ValueError(f'Metadata key missing: {key}')
    if int(metadata['frame_count']) < 225:
        raise ValueError('Frame count too small for freeze frames 150..224')
    freeze_info = fault_video.with_suffix('.json')
    if freeze_info.exists():
        info = json.loads(freeze_info.read_text(encoding='utf-8-sig'))
        for key, expected in [('scenario','freeze'),('severity','low'),('fault_start_frame',150),('fault_end_frame',224),('frame_count',int(metadata['frame_count']))]:
            if info.get(key) != expected:
                raise ValueError(f'Fault sidecar mismatch: {key}={info.get(key)!r}, expected {expected!r}')
    return numbered, fault_video, metadata_path, metadata


def receiver_ready(rtc, driver):
    try:
        data = rtc.fetch_stats(driver)
        stream = rtc.choose_inbound(data)
        video = rtc.choose_remote_video(data)
        return bool(stream and (stream.get('framesDecoded') or 0)>0 and video and
                    (video.get('readyState') or 0)>=2 and
                    (video.get('width') or 0)>0 and (video.get('height') or 0)>0)
    except Exception:
        return False


def preflight_frame(capture, quality, driver, folder: Path, metadata):
    import cv2
    from selenium.webdriver.support.ui import WebDriverWait
    last_error = ''
    for attempt in range(5):
        target = folder/'receiver_preflight.png'
        ok, info = capture.capture_frame(driver, target)
        if ok:
            img = cv2.imread(str(target), cv2.IMREAD_COLOR)
            if img is not None:
                try:
                    idx, confidence, _ = quality.decode_barcode(img, metadata)
                    print(f'  Preflight barcode: frame={idx}, confidence={confidence}', flush=True)
                    return {'frame_index':idx,'confidence':confidence,'width':info.get('width'),
                            'height':info.get('height'),'remote_id':info.get('elementId')}
                except Exception as exc:
                    last_error = str(exc)
        else:
            last_error = str(info)
        time.sleep(0.5)
    raise RuntimeError('Cannot decode the independent frame barcode from Participant_B. '
                       f'Last error: {last_error}. Inspect {folder / "receiver_preflight.png"}')


def collect_one(condition, video_path, metadata, output_root, args, rtc, fault, capture, quality):
    from selenium.webdriver.support.ui import WebDriverWait
    folder = output_root/condition
    folder.mkdir(exist_ok=False)
    rows, stats_rows = [], []
    drivers = {}
    observer_started = False
    rtc.start_participant_original = rtc.start_participant_original if hasattr(rtc,'start_participant_original') else rtc.start_participant
    rtc.start_participant = fault.sender_aware_participant_starter(rtc, video_path)
    session = {'condition':condition, 'video':str(video_path), 'duration_s':args.duration,
               'warmup_s':args.warmup,'status':'STARTING'}
    print(f'\n===== {condition.upper()} =====',flush=True)
    print('  Sender video:',video_path,flush=True)
    try:
        print('  Participant_A starting...',flush=True)
        drivers['A'] = rtc.start_participant('Participant_A')
        print('  Participant_B starting...',flush=True)
        drivers['B'] = rtc.start_participant('Participant_B')
        b = drivers['B']
        print('  Waiting for receiver WebRTC video...',flush=True)
        WebDriverWait(b, 75, poll_frequency=2).until(lambda d: receiver_ready(rtc,d))
        if args.warmup:
            print(f'  Warmup: {args.warmup}s',flush=True)
            time.sleep(args.warmup)
        session['preflight'] = preflight_frame(capture,quality,b,folder,metadata)
        result = b.execute_script(START_JS, {'metadata':metadata})
        if not result or not result.get('ok'):
            raise RuntimeError(f'Could not start requestVideoFrameCallback observer: {result}')
        session['observer'] = result
        observer_started = True
        print('  Receiver observer started:',result,flush=True)
        t0 = time.monotonic()
        next_stats = t0
        while True:
            elapsed = time.monotonic()-t0
            if elapsed >= args.duration:
                break
            time.sleep(min(args.poll, max(0.05,args.duration-elapsed)))
            data = b.execute_script(DRAIN_JS)
            if not data or not data.get('ok'):
                raise RuntimeError(f'Observer drain failed: {data}')
            rows.extend({'condition':condition,**r} for r in data.get('rows',[]))
            if data.get('error') or data.get('overflow') or not data.get('connected', True):
                raise RuntimeError(f'Observer error/overflow/disconnected: {data.get("error")}, overflow={data.get("overflow")}, connected={data.get("connected")}')
            if time.monotonic()>=next_stats:
                try:
                    s = rtc.fetch_stats(b)
                    inbound = rtc.choose_inbound(s)
                    remote = rtc.choose_remote_video(s)
                    stats_rows.append({'elapsed_s':round(time.monotonic()-t0,3),
                        'inbound_id':inbound.get('id') if inbound else '',
                        'frames_decoded':inbound.get('framesDecoded') if inbound else '',
                        'frames_received':inbound.get('framesReceived') if inbound else '',
                        'freeze_count':inbound.get('freezeCount') if inbound else '',
                        'total_freezes_duration_s':inbound.get('totalFreezesDuration') if inbound else '',
                        'packets_lost':inbound.get('packetsLost') if inbound else '',
                        'reported_fps':inbound.get('framesPerSecond') if inbound else '',
                        'remote_video_id':remote.get('id') if remote else '',
                        'remote_video_track_ids':';'.join(remote.get('srcVideoTrackIds') or []) if remote else '',
                        'playback_total_frames':remote.get('totalVideoFrames') if remote else ''})
                except Exception as exc:
                    stats_rows.append({'elapsed_s':round(time.monotonic()-t0,3),'error':repr(exc)})
                next_stats=time.monotonic()+5
            if len(rows) and len(rows)%100<40:
                print(f'  Captured callbacks: {len(rows)}',flush=True)
            # Persist incrementally, including if the next browser call fails.
            csv_save(folder/'receiver_frames.csv',rows,CSV_FIELDS)
        data = b.execute_script(STOP_JS)
        observer_started=False
        if not data or not data.get('ok'):
            raise RuntimeError(f'Observer stop failed: {data}')
        rows.extend({'condition':condition,**r} for r in data.get('rows',[]))
        if data.get('error') or data.get('overflow'):
            raise RuntimeError(f'Observer error/overflow: {data.get("error")}, overflow={data.get("overflow")}')
        session['observer_total_callbacks'] = data['count']
        session['status']='COMPLETE'
    except Exception as exc:
        session['status']='ERROR'
        session['error']=repr(exc)
        print('  SESSION ERROR:',repr(exc),flush=True)
        raise
    finally:
        if observer_started and 'B' in drivers:
            try:
                data=drivers['B'].execute_script(STOP_JS)
                if data and data.get('ok'):
                    rows.extend({'condition':condition,**r} for r in data.get('rows',[]))
            except Exception:
                pass
        csv_save(folder/'receiver_frames.csv',rows,CSV_FIELDS)
        csv_save(folder/'webrtc_stats.csv',stats_rows,[
            'elapsed_s','inbound_id','frames_decoded','frames_received','freeze_count',
            'total_freezes_duration_s','packets_lost','reported_fps','remote_video_id',
            'remote_video_track_ids','playback_total_frames','error'])
        session['received_callback_rows']=len(rows)
        (folder/'session.json').write_text(json.dumps(session,indent=2,ensure_ascii=False),encoding='utf-8')
        for d in drivers.values():
            try: d.quit()
            except Exception: pass
    print(f'  {condition}: {len(rows)} receiver frames observed.',flush=True)
    return rows, session


def frame_pair_summary(rows, metadata):
    count=int(metadata['frame_count'])
    all_pairs=[]
    episodes=[]
    current=None
    decoded=sum(isinstance(r.get('frame_index'),int) for r in rows)
    for i in range(1,len(rows)):
        a,b=rows[i-1],rows[i]
        ia,ib=a.get('frame_index'),b.get('frame_index')
        dt=b.get('callback_gap_ms')
        mae=b.get('scene_mae')
        if not isinstance(ia,int) or not isinstance(ib,int) or not isinstance(dt,(int,float)) or not isinstance(mae,(int,float)):
            valid=False
        else:
            jump=(ib-ia)%count
            valid=(0<jump<=20 and 0<dt<=350 and
                   a.get('video_width')==b.get('video_width') and
                   a.get('video_height')==b.get('video_height'))
        inside=valid and 150<=ia<=224 and 150<=ib<=224 and ib>ia
        if valid:
            all_pairs.append({'frame_index':ib,'prev_frame_index':ia,'mae':mae,
                'inside':inside,'time_ms':b.get('relative_ms'),'dt_ms':dt})
        if inside:
            if current is None:
                current={'start_ms':a.get('relative_ms'),'end_ms':b.get('relative_ms'),
                         'first_index':ia,'last_index':ib,'pair_count':0,'maes':[]}
            current['end_ms']=b.get('relative_ms')
            current['last_index']=ib
            current['pair_count']+=1
            current['maes'].append(mae)
        elif current is not None:
            episodes.append(current)
            current=None
    if current is not None: episodes.append(current)
    for ep in episodes:
        ep['duration_s']=round((ep['end_ms']-ep['start_ms'])/1000,3)
        ep['source_span_frames']=ep['last_index']-ep['first_index']
        ep['median_mae']=median(ep.pop('maes'))
    inside_vals=[p['mae'] for p in all_pairs if p['inside']]
    outside_vals=[p['mae'] for p in all_pairs if not p['inside']]
    return {'callbacks':len(rows),'barcode_decoded':decoded,
            'callback_processing_median_ms':median([r.get('callback_processing_ms') for r in rows]),
            'barcode_success_pct':round(100*decoded/len(rows),2) if rows else 0,
            'valid_advancing_pairs':len(all_pairs),'target_pairs':len(inside_vals),
            'target_median_mae':median(inside_vals),'target_p10_mae':pct(inside_vals,10),
            'target_p90_mae':pct(inside_vals,90),
            'other_median_mae':median(outside_vals),'other_p90_mae':pct(outside_vals,90),
            'episodes':episodes,'pairs':all_pairs}


def create_svg(path, pair_data):
    width,height=1200,520
    left,right,top,bottom=85,35,50,70
    xspan=width-left-right; yspan=height-top-bottom
    binned={}
    all_y=[]
    for cond,pairs in pair_data.items():
        groups=defaultdict(list)
        for p in pairs:
            groups[int(p['frame_index'])//5].append(p['mae'])
        binned[cond]={k:statistics.median(v) for k,v in groups.items()}
        all_y.extend(binned[cond].values())
    ymax=max(1.0,pct(all_y,98)*1.15) if all_y else 1.0
    xmax=504
    x=lambda i:left+i/xmax*xspan
    y=lambda v:top+yspan-min(v,ymax)/ymax*yspan
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
      '<rect width="100%" height="100%" fill="white"/>',
      '<text x="85" y="28" font-family="Arial" font-size="21" font-weight="bold">EXP04 Receiver-side scene MAE by source frame</text>',
      f'<rect x="{x(150):.1f}" y="{top}" width="{x(225)-x(150):.1f}" height="{yspan}" fill="#ffe9bf"/>']
    for t in range(6):
        yy=top+yspan*t/5
        value=ymax*(1-t/5)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    for i in range(0,505,50):
        parts.append(f'<text x="{x(i):.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="12">{i}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#555"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="14">Decoded source frame index (0..503), 5-frame bins</text>')
    parts.append(f'<text x="25" y="{height/2:.1f}" transform="rotate(-90 25 {height/2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">Adjacent scene MAE (0..255)</text>')
    for cond,color in [('baseline','#2774ad'),('freeze_low','#c74343')]:
        points=binned.get(cond,{})
        # Disconnected bins are rendered as separate line segments.
        keys=sorted(points)
        segments=[]; segment=[]; last=None
        for k in keys:
            if last is not None and k-last>1 and segment:
                segments.append(segment); segment=[]
            segment.append(f'{x(k*5+2.5):.1f},{y(points[k]):.1f}')
            last=k
        if segment: segments.append(segment)
        for segment in segments:
            points_attr = ' '.join(segment)
            parts.append(f'<polyline points="{points_attr}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append(f'<text x="{x(150)+6:.1f}" y="{top+20}" font-family="Arial" font-size="13">Target 150..224</text>')
    parts.append('<rect x="850" y="45" width="12" height="12" fill="#2774ad"/>')
    parts.append('<text x="868" y="56" font-family="Arial" font-size="13">Baseline</text>')
    parts.append('<rect x="950" y="45" width="12" height="12" fill="#c74343"/>')
    parts.append('<text x="968" y="56" font-family="Arial" font-size="13">Freeze/Low</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def verdict(baseline, freeze):
    if not baseline or not freeze:
        return 'INCOMPLETE', 'Both baseline and freeze/low receiver sessions are required.'
    if baseline['barcode_success_pct']<80 or freeze['barcode_success_pct']<80:
        return 'INCONCLUSIVE', 'Fewer than 80% of video callbacks have valid independent frame barcodes.'
    bm=baseline['target_median_mae']; fm=freeze['target_median_mae']
    if bm is None or fm is None or baseline['target_pairs']<15 or freeze['target_pairs']<15:
        return 'INCONCLUSIVE', 'Too few valid advancing frame pairs in the target source window.'
    substantial=[ep for ep in freeze['episodes'] if ep['duration_s']>=1.0 and
                 ep['source_span_frames']>=40 and ep['pair_count']>=10]
    if bm>=4*max(fm,0.1) and len(substantial)>=2:
        return 'RECEIVER CONTENT HOLD SUPPORTED', (
            f'Freeze median target MAE={fm:.4f} versus baseline={bm:.4f} '
            f'(ratio={fm/bm:.3f}); {len(substantial)} receiver episodes '
            'lasted at least 1.0 s while source frame IDs advanced by at least 40.')
    return 'INCONCLUSIVE', (f'Target median MAE baseline={bm:.4f}, freeze={fm:.4f}; '
            f'episodes >=1.0 s with >=40 source-frame advancement: {len(substantial)}. '
            'Inspect the frame trace, network stats, and episode durations before interpreting.')


def write_report(folder, summaries, sessions, metadata):
    b=summaries.get('baseline'); f=summaries.get('freeze_low')
    v,reason=verdict(b,f)
    clean={}
    for key,data in summaries.items():
        clean[key]={k:val for k,val in data.items() if k!='pairs'}
    result={'verdict':v,'explanation':reason,'source_fps':metadata['fps'],
            'source_frame_count':metadata['frame_count'],
            'fault_window_frames_inclusive':[150,224],
            'fault_duration_s':75/float(metadata['fps']),
            'method':'Participant_B HTMLVideoElement requestVideoFrameCallback; 320x180 grayscale scene MAE excluding the overlay rectangle; independently decoded 24-bit frame barcode at 640x360; WebRTC getStats as secondary evidence',
            'thresholds':'Descriptive predeclared screening: >=80% valid barcodes, >=15 valid advancing pairs per condition, baseline median >=4x freeze median, >=2 episodes each >=1.0s with >=40 source-frame advancement and >=10 advancing pairs.',
            'limitations':['No full-resolution received video is saved; scene MAE is downscaled to 320x180.',
                'Only frames actually presented by the browser are observed; missing/dropped frames cannot be reconstructed.',
                'Frame counter overlay is excluded from scene MAE, but decoded to verify frame progression.',
                'WebRTC freezeCount is a transport/playback statistic, not an indicator of source-content freeze.',
                'A dedicated validation session does not retroactively establish the behavior of every previous pilot session.',
                'The observed duration may differ from the nominal 1.5s due to receiver frame rate and delivery delays.'],
            'sessions':sessions,'summaries':clean}
    (folder/'temporal_validation_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['EXP04 SHIELDS – RECEIVER-SIDE TEMPORAL VALIDATION',
           '='*60, f'VERDICT: {v}',reason,'',
           'Target source window: frames 150..224 inclusive; nominal duration 1.5 s',
           'Measurement: requestVideoFrameCallback on Participant_B; scene MAE excludes frame-counter overlay',
           '']
    for condition in ('baseline','freeze_low'):
        d=summaries.get(condition)
        if not d: continue
        lines.extend([condition.upper(),f'  Receiver callback rows: {d["callbacks"]}',
            f'  Valid barcode frames: {d["barcode_decoded"]} ({d["barcode_success_pct"]}%)',
            f'  Median callback processing: {d["callback_processing_median_ms"]} ms',
            f'  Valid advancing pairs: {d["valid_advancing_pairs"]}',
            f'  Target-window pairs: {d["target_pairs"]}',
            f'  Target-window median MAE: {d["target_median_mae"]}',
            f'  Target-window p10/p90 MAE: {d["target_p10_mae"]} / {d["target_p90_mae"]}',
            f'  Other-window median MAE: {d["other_median_mae"]}',
            f'  Target-window episodes: {len(d["episodes"])}'])
        for i,ep in enumerate(d['episodes'],1):
            lines.append(f'    episode {i}: frames {ep["first_index"]}..{ep["last_index"]}, '
                         f'duration={ep["duration_s"]}s, pairs={ep["pair_count"]}, '
                         f'median MAE={ep["median_mae"]}')
        lines.append('')
    lines.extend(['CAVEATS:',*('  - '+x for x in result['limitations'])])
    (folder/'temporal_validation_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if summaries:
        create_svg(folder/'temporal_mae.svg',{k:v['pairs'] for k,v in summaries.items()})
    print('\n'+'='*65,flush=True)
    print('VERDICT:',v,flush=True)
    print(reason,flush=True)
    print('Report:',folder/'temporal_validation_report.txt',flush=True)
    print('Detailed JSON:',folder/'temporal_validation_report.json',flush=True)
    print('Plot:',folder/'temporal_mae.svg',flush=True)


def main():
    parser=argparse.ArgumentParser(description='EXP04 Shields receiver-side temporal validation')
    parser.add_argument('--duration',type=int,default=35,help='Seconds of receiver frame callbacks per session (default 35)')
    parser.add_argument('--warmup',type=int,default=5,help='Seconds after receiver video is ready (default 5)')
    parser.add_argument('--poll',type=float,default=2.0,help='Seconds between browser callback drains (default 2)')
    parser.add_argument('--only',choices=('both','baseline','freeze_low'),default='both')
    parser.add_argument('--numbered',help='Full path to baseline numbered Shields Y4M')
    parser.add_argument('--fault-video',help='Full path to freeze/low Shields Y4M')
    parser.add_argument('--metadata',help='Full path to numbered Shields metadata JSON')
    parser.add_argument('--check-only',action='store_true',help='Verify input files without starting Docker/Jitsi')
    args=parser.parse_args()
    if args.duration<15 or args.duration>180 or args.warmup<0 or args.poll<=0:
        parser.error('Require 15<=duration<=180, warmup>=0, poll>0')
    numbered,freeze,metadata_path,metadata=resolve_inputs(args)
    for name in ('05_jitsi_video_capture.py','08_jitsi_synchronized_quality.py',
                 '09_jitsi_webrtc_stats.py','11_jitsi_fault_injection_v2.py'):
        if not (ROOT/name).is_file():
            raise FileNotFoundError('Existing experiment module not found: '+str(ROOT/name))
    if args.check_only:
        print('INPUT CHECK PASSED')
        print('Baseline:',numbered)
        print('Freeze/Low:',freeze)
        print('Metadata:',metadata_path)
        print('Frame count:',metadata['frame_count'],'FPS:',metadata['fps'])
        return
    rtc=load_module('exp04_temporal_rtc09',ROOT/'09_jitsi_webrtc_stats.py')
    fault=load_module('exp04_temporal_fault11',ROOT/'11_jitsi_fault_injection_v2.py')
    capture=load_module('exp04_temporal_capture05',ROOT/'05_jitsi_video_capture.py')
    quality=load_module('exp04_temporal_quality08',ROOT/'08_jitsi_synchronized_quality.py')
    out=ROOT/'captures'/('exp04_temporal_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    print('EXP04 TEMPORAL VALIDATION',flush=True)
    print('Output:',out,flush=True)
    print('Baseline:',numbered,flush=True)
    print('Freeze/Low:',freeze,flush=True)
    print('Metadata:',metadata_path,flush=True)
    (out/'configuration.json').write_text(json.dumps({'numbered':str(numbered),
        'freeze':str(freeze),'metadata':str(metadata_path),
        'duration':args.duration,'warmup':args.warmup,'only':args.only},indent=2),encoding='utf-8')
    conditions=[('baseline',numbered),('freeze_low',freeze)] if args.only=='both' else [
        ('baseline',numbered) if args.only=='baseline' else ('freeze_low',freeze)]
    summaries={}; sessions={}
    try:
        for i,(name,video) in enumerate(conditions):
            if i:
                print('\nWaiting 10 seconds for previous Jitsi sessions to close...',flush=True)
                time.sleep(10)
            rows,session=collect_one(name,video,metadata,out,args,rtc,fault,capture,quality)
            sessions[name]=session
            summaries[name]=frame_pair_summary(rows,metadata)
            write_report(out,summaries,sessions,metadata)
    except Exception:
        (out/'ERROR_TRACEBACK.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('\nA session failed. Existing captured rows and diagnostics are preserved:',out,flush=True)
        if summaries:
            write_report(out,summaries,sessions,metadata)
        raise
    print('\nDone. No original experiment files or past results were modified.',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('\nERROR:',str(exc),file=sys.stderr,flush=True)
        sys.exit(1)
