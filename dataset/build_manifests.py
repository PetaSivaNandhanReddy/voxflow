import re
import os
import json
import time
import pandas as pd
import numpy as np

def build_v2_manifests():
    print("=== VOXFLOW V2: BUILDING FINAL DATASET MANIFESTS ===")
    t0 = time.time()

    # 1. Load Audio Manifest (32,321 entries)
    audio_manifest = pd.read_csv('v2/dataset/manifests/v2_audio_verified_manifest.csv', dtype={'EpId': str, 'ClipId': str})
    audio_manifest['Show'] = audio_manifest['Show'].astype(str).str.strip()
    audio_manifest['EpId'] = audio_manifest['EpId'].astype(str).str.strip()
    audio_manifest['ClipId'] = audio_manifest['ClipId'].astype(str).str.strip()
    audio_manifest['key'] = audio_manifest['Show'] + '_' + audio_manifest['EpId'] + '_' + audio_manifest['ClipId']

    # 2. Load Metadata
    sep_df = pd.read_csv('v2/dataset/metadata/SEP-28k-Extended_clips.csv', dtype={'EpId': str, 'ClipId': str})
    fb_df = pd.read_csv('v2/dataset/metadata/fluencybank_labels.csv', dtype={'EpId': str, 'ClipId': str})
    fb_ep = pd.read_csv('v2/dataset/metadata/fluencybank_episodes.csv', header=None, names=['show', 'ep_id_raw', 'url', 'Show', 'EpId'])

    sep_df['Show'] = sep_df['Show'].astype(str).str.strip()
    sep_df['EpId'] = sep_df['EpId'].astype(str).str.strip()
    sep_df['ClipId'] = sep_df['ClipId'].astype(str).str.strip()
    sep_df['key'] = sep_df['Show'] + '_' + sep_df['EpId'] + '_' + sep_df['ClipId']
    sep_df['source'] = 'SEP-28k'

    fb_df['Show'] = fb_df['Show'].astype(str).str.strip()
    fb_df['EpId'] = fb_df['EpId'].astype(str).str.strip()
    fb_df['ClipId'] = fb_df['ClipId'].astype(str).str.strip()
    fb_df['key'] = fb_df['Show'] + '_' + fb_df['EpId'] + '_' + fb_df['ClipId']
    fb_df['source'] = 'FluencyBank'

    # Map FluencyBank speakers by participant ID from session url
    def get_participant_id(session_str):
        m = re.match(r'^(\d+[mf])([a-c])?$', session_str.strip())
        if m:
            return 'FB_' + m.group(1)
        return 'FB_' + session_str.strip()

    fb_ep['norm_ep'] = fb_ep['EpId'].astype(str).str.strip().astype(int)
    fb_ep['session'] = fb_ep['url'].apply(lambda u: u.strip().split('/')[-1].replace('.mp4', ''))
    fb_ep['speaker'] = fb_ep['session'].apply(get_participant_id)
    ep_to_spk = dict(zip(fb_ep['norm_ep'], fb_ep['speaker']))

    fb_df['norm_ep'] = fb_df['EpId'].astype(str).str.strip().astype(int)
    fb_df['speaker'] = fb_df['norm_ep'].map(ep_to_spk)
    fb_df['SEP28k-E'] = 'unassigned'

    # Combine metadata
    meta_cols = ['Show', 'EpId', 'ClipId', 'key', 'source', 'speaker', 'SEP28k-E',
                 'SoundRep', 'WordRep', 'Prolongation', 'Block', 'Interjection', 
                 'NoStutteredWords', 'NaturalPause', 'NoSpeech', 'PoorAudioQuality', 
                 'DifficultToUnderstand', 'Unsure', 'Music']
    
    comb_meta = pd.concat([sep_df[meta_cols], fb_df[meta_cols]], ignore_index=True)

    # 3. Join with Audio Manifest
    merged = pd.merge(comb_meta, audio_manifest[['key', 'archive_path', 'duration', 'sample_rate', 'channels', 'category', 'valid_audio']], on='key')
    print(f"Total merged records: {len(merged)}")
    assert len(merged) == 32321, f"Expected 32321, got {len(merged)}"

    # 4. Exclusions Accounting
    total_raw_metadata = len(merged)
    valid_full_audio = int((merged['category'] == 'VALID').sum())
    excluded_empty = int((merged['category'] == 'EMPTY_HEADER').sum())
    excluded_partial = int((merged['category'] == 'PARTIAL_TRUNCATED').sum())

    # Step 2: Hard Audio Exclusion
    df_valid = merged[merged['category'] == 'VALID'].copy()
    assert len(df_valid) == 31853, f"Expected 31853 valid clips, got {len(df_valid)}"

    # Step 3: Hard Quality Exclusion (PoorAudioQuality >= 2 | NoSpeech >= 2 | Unsure >= 2)
    paq_ge2 = df_valid['PoorAudioQuality'] >= 2
    nospeech_ge2 = df_valid['NoSpeech'] >= 2
    unsure_ge2 = df_valid['Unsure'] >= 2
    hard_quality_mask = paq_ge2 | nospeech_ge2 | unsure_ge2
    quality_exclusions = int(hard_quality_mask.sum())

    df_clean = df_valid[~hard_quality_mask].copy()
    final_usable_count = len(df_clean)
    assert final_usable_count == 30999, f"Expected 30999 clean clips, got {final_usable_count}"

    # 5. Label Assignment (Policy C with Policy A majority vote consensus)
    df_clean['Repetition_max'] = df_clean[['SoundRep', 'WordRep']].max(axis=1)
    df_clean['repetition_label'] = (df_clean['Repetition_max'] >= 2).astype(int)
    df_clean['prolongation_label'] = (df_clean['Prolongation'] >= 2).astype(int)
    df_clean['block_label'] = (df_clean['Block'] >= 2).astype(int)

    df_clean['target_count'] = df_clean['repetition_label'] + df_clean['prolongation_label'] + df_clean['block_label']
    df_clean['multi_label'] = df_clean['target_count'] >= 2

    # Annotation conflicts
    df_clean['annotation_conflict'] = (df_clean['NoStutteredWords'] >= 2) & (df_clean['target_count'] >= 1)

    # Fluent candidate
    df_clean['fluent_candidate'] = (df_clean['target_count'] == 0) & (df_clean['NoStutteredWords'] >= 2) & (~df_clean['annotation_conflict'])

    # Confidence score (max vote / 3.0)
    df_clean['confidence_score'] = round(df_clean[['Repetition_max', 'Prolongation', 'Block']].max(axis=1) / 3.0, 3)

    # Agreement categories
    target_max_vote = df_clean[['Repetition_max', 'Prolongation', 'Block']].max(axis=1)
    vote_0_3 = int((target_max_vote == 0).sum())
    vote_1_3 = int((target_max_vote == 1).sum())
    vote_2_3 = int((target_max_vote == 2).sum())
    vote_3_3 = int((target_max_vote == 3).sum())

    # 6. Speaker-Exclusive Split Assignment
    fb_participants = sorted(list(df_clean[df_clean['Show'] == 'FluencyBank']['speaker'].dropna().unique()))
    np.random.seed(42)
    shuffled_fb = np.random.permutation(fb_participants)
    fb_train_spk = set(shuffled_fb[:17])
    fb_val_spk = set(shuffled_fb[17:21])
    fb_test_spk = set(shuffled_fb[21:])

    def assign_split(row):
        if row['Show'] == 'FluencyBank':
            spk = row['speaker']
            if spk in fb_train_spk:
                return 'train'
            elif spk in fb_val_spk:
                return 'val'
            else:
                return 'test'
        else:
            s = row['SEP28k-E']
            if s == 'dev':
                return 'val'
            return s

    df_clean['split'] = df_clean.apply(assign_split, axis=1)

    # Verify Speaker Overlap
    train_spks = set(df_clean[df_clean['split'] == 'train']['speaker'].dropna())
    val_spks = set(df_clean[df_clean['split'] == 'val']['speaker'].dropna())
    test_spks = set(df_clean[df_clean['split'] == 'test']['speaker'].dropna())

    ov_train_val = len(train_spks & val_spks)
    ov_train_test = len(train_spks & test_spks)
    ov_val_test = len(val_spks & test_spks)
    total_overlap = ov_train_val + ov_train_test + ov_val_test
    assert total_overlap == 0, f"Speaker overlap detected: {ov_train_val}, {ov_train_test}, {ov_val_test}"

    # Export Manifests
    export_cols = [
        'Show', 'EpId', 'ClipId', 'speaker', 'source', 'archive_path',
        'repetition_label', 'prolongation_label', 'block_label',
        'target_count', 'multi_label', 'fluent_candidate', 'annotation_conflict',
        'confidence_score',
        'SoundRep', 'WordRep', 'Prolongation', 'Block', 'Interjection', 
        'NoStutteredWords', 'NaturalPause', 'NoSpeech', 'PoorAudioQuality', 
        'DifficultToUnderstand', 'Unsure', 'Music',
        'duration', 'sample_rate', 'channels', 'split'
    ]

    manifest_dir = "v2/dataset/manifests"
    os.makedirs(manifest_dir, exist_ok=True)

    full_manifest_path = f"{manifest_dir}/v2_full_clean_manifest.csv"
    train_manifest_path = f"{manifest_dir}/v2_train_manifest.csv"
    val_manifest_path = f"{manifest_dir}/v2_val_manifest.csv"
    test_manifest_path = f"{manifest_dir}/v2_test_manifest.csv"

    df_clean[export_cols].to_csv(full_manifest_path, index=False)
    df_clean[df_clean['split'] == 'train'][export_cols].to_csv(train_manifest_path, index=False)
    df_clean[df_clean['split'] == 'val'][export_cols].to_csv(val_manifest_path, index=False)
    df_clean[df_clean['split'] == 'test'][export_cols].to_csv(test_manifest_path, index=False)

    print(f"Generated {full_manifest_path}: {len(df_clean)} rows")
    print(f"Generated {train_manifest_path}: {len(df_clean[df_clean['split'] == 'train'])} rows")
    print(f"Generated {val_manifest_path}: {len(df_clean[df_clean['split'] == 'val'])} rows")
    print(f"Generated {test_manifest_path}: {len(df_clean[df_clean['split'] == 'test'])} rows")

    # Aggregate Statistics
    train_df = df_clean[df_clean['split'] == 'train']
    val_df = df_clean[df_clean['split'] == 'val']
    test_df = df_clean[df_clean['split'] == 'test']

    stats = {
        'corpus_funnel': {
            'raw_metadata': total_raw_metadata,
            'valid_full_audio': valid_full_audio,
            'excluded_empty': excluded_empty,
            'excluded_partial': excluded_partial,
            'quality_exclusions': quality_exclusions,
            'final_usable_dataset': final_usable_count
        },
        'targets_summary': {
            'repetition': int(df_clean['repetition_label'].sum()),
            'prolongation': int(df_clean['prolongation_label'].sum()),
            'block': int(df_clean['block_label'].sum()),
            'interjection': int((df_clean['Interjection'] >= 2).sum()),
            'multi_label': int(df_clean['multi_label'].sum()),
            'no_target': int((df_clean['target_count'] == 0).sum()),
            'fluent_candidates': int(df_clean['fluent_candidate'].sum()),
            'annotation_conflicts': int(df_clean['annotation_conflict'].sum())
        },
        'multi_label_combinations': {
            'rep_only_100': int(((df_clean['repetition_label'] == 1) & (df_clean['prolongation_label'] == 0) & (df_clean['block_label'] == 0)).sum()),
            'pro_only_010': int(((df_clean['repetition_label'] == 0) & (df_clean['prolongation_label'] == 1) & (df_clean['block_label'] == 0)).sum()),
            'blk_only_001': int(((df_clean['repetition_label'] == 0) & (df_clean['prolongation_label'] == 0) & (df_clean['block_label'] == 1)).sum()),
            'rep_pro_110': int(((df_clean['repetition_label'] == 1) & (df_clean['prolongation_label'] == 1) & (df_clean['block_label'] == 0)).sum()),
            'rep_blk_101': int(((df_clean['repetition_label'] == 1) & (df_clean['prolongation_label'] == 0) & (df_clean['block_label'] == 1)).sum()),
            'pro_blk_011': int(((df_clean['repetition_label'] == 0) & (df_clean['prolongation_label'] == 1) & (df_clean['block_label'] == 1)).sum()),
            'all_three_111': int(((df_clean['repetition_label'] == 1) & (df_clean['prolongation_label'] == 1) & (df_clean['block_label'] == 1)).sum()),
            'no_target_000': int((df_clean['target_count'] == 0).sum())
        },
        'agreement_distribution': {
            '0_of_3': vote_0_3,
            '1_of_3': vote_1_3,
            '2_of_3': vote_2_3,
            '3_of_3': vote_3_3
        },
        'split_distribution': {
            'train': {
                'clips': len(train_df),
                'pct': round(len(train_df) / final_usable_count * 100, 2),
                'speakers': len(train_spks),
                'repetition': int(train_df['repetition_label'].sum()),
                'prolongation': int(train_df['prolongation_label'].sum()),
                'block': int(train_df['block_label'].sum()),
                'multi_label': int(train_df['multi_label'].sum()),
                'no_target': int((train_df['target_count'] == 0).sum()),
                'fluent_candidates': int(train_df['fluent_candidate'].sum())
            },
            'validation': {
                'clips': len(val_df),
                'pct': round(len(val_df) / final_usable_count * 100, 2),
                'speakers': len(val_spks),
                'repetition': int(val_df['repetition_label'].sum()),
                'prolongation': int(val_df['prolongation_label'].sum()),
                'block': int(val_df['block_label'].sum()),
                'multi_label': int(val_df['multi_label'].sum()),
                'no_target': int((val_df['target_count'] == 0).sum()),
                'fluent_candidates': int(val_df['fluent_candidate'].sum())
            },
            'test': {
                'clips': len(test_df),
                'pct': round(len(test_df) / final_usable_count * 100, 2),
                'speakers': len(test_spks),
                'repetition': int(test_df['repetition_label'].sum()),
                'prolongation': int(test_df['prolongation_label'].sum()),
                'block': int(test_df['block_label'].sum()),
                'multi_label': int(test_df['multi_label'].sum()),
                'no_target': int((test_df['target_count'] == 0).sum()),
                'fluent_candidates': int(test_df['fluent_candidate'].sum())
            }
        },
        'speaker_overlap': {
            'train_val': ov_train_val,
            'train_test': ov_train_test,
            'val_test': ov_val_test,
            'total_overlap': total_overlap
        }
    }

    # Save JSON Report
    json_path = 'v2/audit/v2_dataset_lock.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {json_path}")

    # Generate Markdown Report
    md_path = 'v2/audit/V2_DATASET_LOCK_REPORT.md'
    with open(md_path, 'w') as f:
        f.write("# VoxFlow V2 — Dataset Lock & Manifest Finalization Report\n\n")
        f.write("**Date:** September 19, 2026  \n")
        f.write("**Status:** LOCKED FOR PRODUCTION ML (Step 3 Complete)  \n")
        f.write(f"**Canonical Full Manifest:** [`v2/dataset/manifests/v2_full_clean_manifest.csv`](file://{os.path.abspath(full_manifest_path)})  \n")
        f.write(f"**Train Manifest:** [`v2/dataset/manifests/v2_train_manifest.csv`](file://{os.path.abspath(train_manifest_path)})  \n")
        f.write(f"**Validation Manifest:** [`v2/dataset/manifests/v2_val_manifest.csv`](file://{os.path.abspath(val_manifest_path)})  \n")
        f.write(f"**Test Manifest:** [`v2/dataset/manifests/v2_test_manifest.csv`](file://{os.path.abspath(test_manifest_path)})  \n")
        f.write(f"**Machine-Readable Lock Data:** [`v2/audit/v2_dataset_lock.json`](file://{os.path.abspath(json_path)})  \n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary & Funnel Metrics\n\n")
        f.write("```text\n")
        f.write(f"RAW METADATA:           {total_raw_metadata:,}\n")
        f.write(f"VALID FULL AUDIO:       {valid_full_audio:,}\n")
        f.write(f"EXCLUDED EMPTY:         {excluded_empty:,}\n")
        f.write(f"EXCLUDED PARTIAL:       {excluded_partial:,}\n")
        f.write(f"QUALITY EXCLUSIONS:     {quality_exclusions:,}\n")
        f.write(f"FINAL USABLE DATASET:   {final_usable_count:,}\n")
        f.write("```\n\n")
        f.write("### Target Event Distribution (Final Usable Dataset)\n\n")
        f.write("```text\n")
        f.write(f"REPETITION:             {stats['targets_summary']['repetition']:,} ({stats['targets_summary']['repetition']/final_usable_count*100:.2f}%)\n")
        f.write(f"PROLONGATION:           {stats['targets_summary']['prolongation']:,} ({stats['targets_summary']['prolongation']/final_usable_count*100:.2f}%)\n")
        f.write(f"BLOCK:                  {stats['targets_summary']['block']:,} ({stats['targets_summary']['block']/final_usable_count*100:.2f}%)\n")
        f.write(f"INTERJECTION:           {stats['targets_summary']['interjection']:,} ({stats['targets_summary']['interjection']/final_usable_count*100:.2f}% — non-target)\n")
        f.write(f"MULTI-LABEL:            {stats['targets_summary']['multi_label']:,} ({stats['targets_summary']['multi_label']/final_usable_count*100:.2f}%)\n")
        f.write(f"NO-TARGET:              {stats['targets_summary']['no_target']:,} ({stats['targets_summary']['no_target']/final_usable_count*100:.2f}%)\n")
        f.write(f"FLUENT CANDIDATES:      {stats['targets_summary']['fluent_candidates']:,} ({stats['targets_summary']['fluent_candidates']/final_usable_count*100:.2f}%)\n")
        f.write("```\n\n")
        f.write("### Annotator Agreement Levels\n\n")
        f.write("```text\n")
        f.write(f"0/3 Annotators:         {vote_0_3:,} ({vote_0_3/final_usable_count*100:.2f}%)\n")
        f.write(f"1/3 Annotator:          {vote_1_3:,} ({vote_1_3/final_usable_count*100:.2f}%)\n")
        f.write(f"2/3 Annotators:         {vote_2_3:,} ({vote_2_3/final_usable_count*100:.2f}%)\n")
        f.write(f"3/3 Annotators:         {vote_3_3:,} ({vote_3_3/final_usable_count*100:.2f}%)\n")
        f.write("```\n\n")
        f.write("### Speaker-Exclusive Split Summary\n\n")
        f.write("```text\n")
        f.write(f"TRAIN:                  {len(train_df):,} ({len(train_df)/final_usable_count*100:.2f}%)\n")
        f.write(f"VALIDATION:             {len(val_df):,} ({len(val_df)/final_usable_count*100:.2f}%)\n")
        f.write(f"TEST:                   {len(test_df):,} ({len(test_df)/final_usable_count*100:.2f}%)\n\n")
        f.write(f"TRAIN SPEAKERS:         {len(train_spks)}\n")
        f.write(f"VALIDATION SPEAKERS:    {len(val_spks)}\n")
        f.write(f"TEST SPEAKERS:          {len(test_spks)}\n\n")
        f.write(f"SPEAKER OVERLAP:        0 (Strictly zero speaker overlap verified across all splits)\n")
        f.write("```\n\n")
        f.write("---\n\n")
        f.write("## 2. Multi-Label Combinations Breakdown\n\n")
        f.write("| Code | Target Combination | Count | Percentage | Clinical Description |\n")
        f.write("| :---: | :--- | :---: | :---: | :--- |\n")
        f.write(f"| `[0, 0, 0]` | No Target Event | {stats['multi_label_combinations']['no_target_000']:,} | {stats['multi_label_combinations']['no_target_000']/final_usable_count*100:.2f}% | Fluent speech, fillers, or pauses |\n")
        f.write(f"| `[1, 0, 0]` | Repetition Only | {stats['multi_label_combinations']['rep_only_100']:,} | {stats['multi_label_combinations']['rep_only_100']/final_usable_count*100:.2f}% | Sound or word repetitions without block/prolongation |\n")
        f.write(f"| `[0, 1, 0]` | Prolongation Only | {stats['multi_label_combinations']['pro_only_010']:,} | {stats['multi_label_combinations']['pro_only_010']/final_usable_count*100:.2f}% | Sound prolongations without repetition/block |\n")
        f.write(f"| `[0, 0, 1]` | Block Only | {stats['multi_label_combinations']['blk_only_001']:,} | {stats['multi_label_combinations']['blk_only_001']/final_usable_count*100:.2f}% | Postural fixations/silent blocks |\n")
        f.write(f"| `[1, 1, 0]` | Repetition + Prolongation | {stats['multi_label_combinations']['rep_pro_110']:,} | {stats['multi_label_combinations']['rep_pro_110']/final_usable_count*100:.2f}% | Combined clonic and tonic disfluency |\n")
        f.write(f"| `[1, 0, 1]` | Repetition + Block | {stats['multi_label_combinations']['rep_blk_101']:,} | {stats['multi_label_combinations']['rep_blk_101']/final_usable_count*100:.2f}% | Block release transitioning into repetition |\n")
        f.write(f"| `[0, 1, 1]` | Prolongation + Block | {stats['multi_label_combinations']['pro_blk_011']:,} | {stats['multi_label_combinations']['pro_blk_011']/final_usable_count*100:.2f}% | Block terminating in prolonged sound |\n")
        f.write(f"| `[1, 1, 1]` | All Three Targets | {stats['multi_label_combinations']['all_three_111']:,} | {stats['multi_label_combinations']['all_three_111']/final_usable_count*100:.2f}% | Severe complex disfluency episode |\n")
        f.write(f"| — | **Total Multi-Label (≥2)** | **{stats['targets_summary']['multi_label']:,}** | **{stats['targets_summary']['multi_label']/final_usable_count*100:.2f}%** | **Preserved multi-label events** |\n\n")
        f.write("---\n\n")
        f.write("## 3. Split Class Coverage Matrix\n\n")
        f.write("| Split | Total Clips | Speakers | Repetition | Prolongation | Block | Multi-Label (≥2) | No-Target | Fluent Candidate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        tr = stats['split_distribution']['train']
        va = stats['split_distribution']['validation']
        te = stats['split_distribution']['test']
        f.write(f"| **TRAIN** | {tr['clips']:,} ({tr['pct']}%) | {tr['speakers']} | {tr['repetition']:,} ({tr['repetition']/tr['clips']*100:.2f}%) | {tr['prolongation']:,} ({tr['prolongation']/tr['clips']*100:.2f}%) | {tr['block']:,} ({tr['block']/tr['clips']*100:.2f}%) | {tr['multi_label']:,} ({tr['multi_label']/tr['clips']*100:.2f}%) | {tr['no_target']:,} ({tr['no_target']/tr['clips']*100:.2f}%) | {tr['fluent_candidates']:,} ({tr['fluent_candidates']/tr['clips']*100:.2f}%) |\n")
        f.write(f"| **VAL** | {va['clips']:,} ({va['pct']}%) | {va['speakers']} | {va['repetition']:,} ({va['repetition']/va['clips']*100:.2f}%) | {va['prolongation']:,} ({va['prolongation']/va['clips']*100:.2f}%) | {va['block']:,} ({va['block']/va['clips']*100:.2f}%) | {va['multi_label']:,} ({va['multi_label']/va['clips']*100:.2f}%) | {va['no_target']:,} ({va['no_target']/va['clips']*100:.2f}%) | {va['fluent_candidates']:,} ({va['fluent_candidates']/va['clips']*100:.2f}%) |\n")
        f.write(f"| **TEST** | {te['clips']:,} ({te['pct']}%) | {te['speakers']} | {te['repetition']:,} ({te['repetition']/te['clips']*100:.2f}%) | {te['prolongation']:,} ({te['prolongation']/te['clips']*100:.2f}%) | {te['block']:,} ({te['block']/te['clips']*100:.2f}%) | {te['multi_label']:,} ({te['multi_label']/te['clips']*100:.2f}%) | {te['no_target']:,} ({te['no_target']/te['clips']*100:.2f}%) | {te['fluent_candidates']:,} ({te['fluent_candidates']/te['clips']*100:.2f}%) |\n\n")
        f.write("---\n\n")
        f.write("## 4. Why These Filtering Rules Were Chosen\n\n")
        f.write("1. **Hard Audio Exclusions (468 clips):**\n")
        f.write("   - **413 Empty Header Files:** 44-byte RIFF headers contain 0 audio samples (resulting from Apple extraction timestamps exceeding raw episode length). These cannot be used for acoustic modeling.\n")
        f.write("   - **55 Partial Files:** Incomplete boundary clips (< 3.0 seconds) that would introduce uneven duration artifacts.\n")
        f.write("2. **Acoustic Quality Exclusions (854 clips):**\n")
        f.write("   - Clips with majority annotator flags for `PoorAudioQuality >= 2` (577 clips), `NoSpeech >= 2` (270 clips), or `Unsure >= 2` (38 clips) are excluded to prevent training models on microphone clipping, static, or dead silence.\n")
        f.write("3. **Preservation of `DifficultToUnderstand >= 2`:**\n")
        f.write("   - Analysis revealed that 466 clips marked `DifficultToUnderstand >= 2` contain confirmed clinical stuttering events (including 226 severe blocks and 78 multi-label events). Excluding this flag would systematically discard severe clinical disfluencies, biassing the model toward mild stuttering.\n")
        f.write("4. **Policy C Multi-Label Consensus (≥ 2/3):**\n")
        f.write("   - Unanimous voting (Policy B) causes severe class starvation, reducing Block samples by 84% (from 3,683 down to 576) and destroying 97% of multi-label instances. Majority voting ($\\ge 2/3$) provides strong consensus while preserving 11,235 positive stuttering instances and 1,172 multi-label co-occurrences.\n\n")
        f.write("---\n\n")
        f.write("## 5. Why This Split Was Chosen\n\n")
        f.write("1. **Strict Speaker Exclusivity:** Zero speaker overlap ($0$ across Train ∩ Val, Train ∩ Test, and Val ∩ Test) prevents acoustic fingerprint leakage and ensures that the model learns disfluency patterns rather than memorizing speaker vocal tracts.\n")
        f.write("2. **Canonical Podcast Partitioning (SEP28k-E):** Keeps the 4 high-volume podcast hosts (Pamela Mertz, Daniele Rossi, Pedro Peña, Peter Reitzes) in Train and partitions the hundreds of guest speakers across Validation and Test. This directly tests out-of-speaker clinical generalization.\n")
        f.write("3. **TalkBank Participant Clustering:** Multiple interview sessions from the same TalkBank participant (e.g. `24fa`, `24fb`, `24fc`) were clustered into unified participant IDs (`FB_24f`) prior to splitting. 17 participants were assigned to Train, 4 to Validation, and 4 to Test (fixed seed 42), preventing session-to-session leakage.\n")
        f.write("4. **Balanced Prevalence:** As shown in the Class Coverage Matrix, the prevalence of Repetition (~18%), Prolongation (~10%), Block (~12%), and Multi-label (~4%) is exceptionally uniform across all three splits.\n")

    print(f"Wrote {md_path}")
    print(f"Dataset locking completed in {time.time()-t0:.2f}s!")

if __name__ == '__main__':
    build_v2_manifests()
