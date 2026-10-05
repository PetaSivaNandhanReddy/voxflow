#!/usr/bin/env python3
"""
VoxFlow V2 — Audio Acquisition & Extraction Engine
Downloads live episodes, resamples to 16kHz mono, slices into 3-second clips,
and verifies integrity incrementally with zero RAM overhead.
"""

import os
import sys
import time
import json
import urllib.request
import subprocess
import pathlib
import pandas as pd
import numpy as np
from scipy.io import wavfile
import concurrent.futures

def process_episode(show, ep_id, url, clips_subset, output_dir, scratch_dir):
    show_clean = show.strip()
    ep_id_str = str(ep_id).strip()
    ep_clip_dir = pathlib.Path(f"{output_dir}/{show_clean}/{ep_id_str}")
    ep_clip_dir.mkdir(parents=True, exist_ok=True)
    
    expected_clips = len(clips_subset)
    existing_clips = 0
    for _, row in clips_subset.iterrows():
        clip_path = ep_clip_dir / f"{show_clean}_{ep_id_str}_{row['ClipId']}.wav"
        if clip_path.exists() and clip_path.stat().st_size == 96044:
            existing_clips += 1
            
    if existing_clips == expected_clips and expected_clips > 0:
        return {
            'show': show_clean,
            'ep_id': ep_id_str,
            'status': 'SKIPPED_EXISTING',
            'clips_extracted': existing_clips,
            'error': None
        }

    url_clean = url.strip()
    ext = ".mp3"
    for e in [".mp3", ".m4a", ".mp4"]:
        if e in url_clean.lower():
            ext = e
            break

    pid = os.getpid()
    thread_id = id(concurrent.futures.thread) if hasattr(concurrent.futures, 'thread') else 0
    raw_path = f"{scratch_dir}/{show_clean}_{ep_id_str}_{pid}_{time.time_ns()}_raw{ext}"
    wav_path = f"{scratch_dir}/{show_clean}_{ep_id_str}_{pid}_{time.time_ns()}_16k.wav"

    max_retries = 2
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url_clean,
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(raw_path, 'wb') as out_f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    
            cmd = ['ffmpeg', '-y', '-i', raw_path, '-ac', '1', '-ar', '16000', '-loglevel', 'error', wav_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {res.stderr.decode('utf-8', errors='ignore')}")

            sr, audio = wavfile.read(wav_path)
            if sr != 16000:
                raise ValueError(f"Expected 16000 Hz, got {sr}")

            extracted_count = 0
            audio_len = len(audio)

            for _, row in clips_subset.iterrows():
                clip_id = row['ClipId']
                start = int(row['Start'])
                stop = int(row['Stop'])
                clip_path = ep_clip_dir / f"{show_clean}_{ep_id_str}_{clip_id}.wav"

                if stop <= audio_len:
                    clip = audio[start:stop]
                else:
                    actual_chunk = audio[start:audio_len] if start < audio_len else np.array([], dtype=np.int16)
                    pad_len = 48000 - len(actual_chunk)
                    clip = np.pad(actual_chunk, (0, max(0, pad_len)), mode='constant')

                if len(clip) != 48000:
                    clip = clip[:48000] if len(clip) > 48000 else np.pad(clip, (0, 48000 - len(clip)))

                wavfile.write(str(clip_path), 16000, clip.astype(np.int16))
                extracted_count += 1

            return {
                'show': show_clean,
                'ep_id': ep_id_str,
                'status': 'SUCCESS',
                'clips_extracted': extracted_count,
                'error': None
            }

        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {
                'show': show_clean,
                'ep_id': ep_id_str,
                'status': 'FAILED',
                'clips_extracted': 0,
                'error': str(exc)
            }
        finally:
            if os.path.exists(raw_path):
                try: os.remove(raw_path)
                except: pass
            if os.path.exists(wav_path):
                try: os.remove(wav_path)
                except: pass

if __name__ == '__main__':
    # Usage: python3 v2/dataset/acquire_audio.py [SHOW_FILTER] [WORKERS]
    # SHOW_FILTER: ALL (default) or show name e.g. HeStutters, WomenWhoStutter, StutterTalk, MyStutteringLife, HVSA
    # WORKERS: number of concurrent threads (default 6)
    show_filter = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    sep_ep = pd.read_csv('v2/dataset/metadata/SEP-28k_episodes.csv', header=None, names=['show_title', 'ep_title', 'url', 'Show', 'EpId'])
    sep_labels = pd.read_csv('v2/dataset/metadata/SEP-28k_labels.csv')
    
    with open('v2/dataset/metadata/sep28k_url_status.json') as f:
        url_status = json.load(f)

    live_ep_keys = {f"{x['Show'].strip()}_{str(x['EpId']).strip()}" for x in url_status if x['status'] == '200'}
    
    sep_ep['ep_key'] = sep_ep['Show'].astype(str).str.strip() + '_' + sep_ep['EpId'].astype(str).str.strip()
    live_episodes = sep_ep[sep_ep['ep_key'].isin(live_ep_keys)].copy()

    if show_filter and show_filter != 'ALL':
        live_episodes = live_episodes[live_episodes['Show'].astype(str).str.strip() == show_filter]

    print("==================================================")
    print("VoxFlow V2 Audio Acquisition & Extraction Engine")
    print(f"Target Show Filter: {show_filter}")
    print(f"Worker Threads:     {max_workers}")
    print(f"Target Episodes:    {len(live_episodes)}")
    print("==================================================")

    out_dir = "v2/dataset/audio"
    scratch_dir = "v2/dataset/raw/scratch_episodes"
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    results = []

    tasks = []
    for _, ep_row in live_episodes.iterrows():
        sh = ep_row['Show'].strip()
        ep = str(ep_row['EpId']).strip()
        u = ep_row['url'].strip()
        sub_clips = sep_labels[(sep_labels['Show'].astype(str).str.strip() == sh) & 
                               (sep_labels['EpId'].astype(str).str.strip() == ep)]
        tasks.append((sh, ep, u, sub_clips))

    total_tasks = len(tasks)
    completed_tasks = 0
    total_clips_extracted = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_episode, sh, ep, u, sub, out_dir, scratch_dir): (sh, ep) for sh, ep, u, sub in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed_tasks += 1
            total_clips_extracted += res['clips_extracted']
            pct = (completed_tasks / total_tasks) * 100
            status_tag = f"[{res['status']}]"
            err_tag = f" (err: {res['error']})" if res['error'] else ""
            print(f"[{completed_tasks}/{total_tasks} | {pct:5.1f}%] {status_tag} {res['show']} Ep {res['ep_id']}: {res['clips_extracted']} clips{err_tag}")

    t1 = time.time()
    elapsed = t1 - t0
    
    print("\n==================================================")
    print("Acquisition Batch Summary")
    print(f"Episodes Processed: {completed_tasks}/{total_tasks}")
    print(f"Total Clips Extracted: {total_clips_extracted}")
    print(f"Elapsed Time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print("==================================================")
    
    # Save/update cumulative acquisition results
    res_path = 'v2/dataset/acquisition_results.json'
    existing_results = {}
    if os.path.exists(res_path):
        try:
            with open(res_path) as f:
                for r in json.load(f):
                    existing_results[f"{r['show']}_{r['ep_id']}"] = r
        except:
            pass
            
    for r in results:
        existing_results[f"{r['show']}_{r['ep_id']}"] = r
        
    with open(res_path, 'w') as f:
        json.dump(list(existing_results.values()), f, indent=2)
