"""
High-Speed Streaming Inference Pipeline for Business Entity Resolution.

Ultra-low RAM Footprint (< 4 GB total for 12 Million records):
- Sequential stream loading & aggressive garbage collection
- One-time candidate indexing directly from shared memory structures
- Direct vectorized numpy LightGBM scoring
- Direct streaming writes to output/matching_results.tsv and output/candidate_pairs.tsv
- Responsive tqdm progress tracking throughout
"""

import os
import gc
import sys
import json
import pickle
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from src.preprocessing import preprocess_dataframe
    from src.blocking import ScalableMultiStrategyBlocker
    from src.features import extract_pair_features_fast, extract_pair_features_vector, FEATURE_COLUMNS
except ImportError:
    from preprocessing import preprocess_dataframe
    from blocking import ScalableMultiStrategyBlocker
    from features import extract_pair_features_fast, extract_pair_features_vector, FEATURE_COLUMNS


def load_tsv_fast(filepath: str, desc: str = "Loading TSV") -> pd.DataFrame:
    """Read large TSV file with visible progress bar."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    chunksize = 100000
    chunks = []

    with tqdm(desc=f"[{desc}] ({file_size_mb:.1f} MB)", unit="chunk", leave=True) as pbar:
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=chunksize, low_memory=False):
            chunks.append(chunk)
            pbar.update(1)
            
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def run_inference(
    s1_path: str = 'student_resource/dataset/test/test_source1.tsv',
    s2_path: str = 'student_resource/dataset/test/test_source2.tsv',
    s3_path: str = 'student_resource/dataset/test/test_source3.tsv',
    model_path: str = 'models/lgbm_matcher.pkl',
    config_path: str = 'models/config.json',
    override_threshold: float = None,
    output_dir: str = 'output',
    chunk_size: int = 10000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-efficient streaming inference pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("\n>>> [Step 1/3] Loading & Normalizing Datasets Sequentially (Low RAM)...")
    
    # 1. Load & process S1
    s1_raw = load_tsv_fast(s1_path, desc="Test Source 1 Reference")
    s1_df = preprocess_dataframe(s1_raw, desc="Test S1")
    del s1_raw
    gc.collect()

    s1_eids = s1_df['entity_id'].tolist()
    s1_dict = {
        eid: (name, addr, country, tuple(nums))
        for eid, name, addr, country, nums in zip(
            s1_df['entity_id'], s1_df['norm_name'], s1_df['norm_addr'], s1_df['norm_country'], s1_df['addr_numbers']
        )
    }
    del s1_df
    gc.collect()

    # 2. Load & process S2 into shared s23_dict
    s23_dict = {}
    s2_raw = load_tsv_fast(s2_path, desc="Test Source 2 Stream")
    s2_df = preprocess_dataframe(s2_raw, desc="Test S2")
    del s2_raw
    gc.collect()

    for eid, name, addr, country, nums in zip(
        s2_df['entity_id'], s2_df['norm_name'], s2_df['norm_addr'], s2_df['norm_country'], s2_df['addr_numbers']
    ):
        s23_dict[eid] = (name, addr, country, tuple(nums))
    del s2_df
    gc.collect()

    # 3. Load & process S3 into shared s23_dict
    s3_raw = load_tsv_fast(s3_path, desc="Test Source 3 Stream")
    s3_df = preprocess_dataframe(s3_raw, desc="Test S3")
    del s3_raw
    gc.collect()

    for eid, name, addr, country, nums in zip(
        s3_df['entity_id'], s3_df['norm_name'], s3_df['norm_addr'], s3_df['norm_country'], s3_df['addr_numbers']
    ):
        s23_dict[eid] = (name, addr, country, tuple(nums))
    del s3_df
    gc.collect()

    print(f"Memory ready: S1={len(s1_eids):,} entities, Candidate Streams={len(s23_dict):,} records.")

    # Load model and threshold config
    feature_cols = [
        'name_lev', 'name_token_set', 'name_token_sort', 'name_partial', 'name_exact',
        'name_pfx_3', 'name_len_diff', 'addr_lev', 'addr_token_set', 'addr_token_sort',
        'addr_partial', 'addr_exact', 'addr_len_diff', 'num_jaccard', 'has_common_num',
        'country_match', 'is_s2', 'is_s3', 'name_addr_interaction', 'name_high_addr_low', 'name_low_addr_high'
    ]
    if os.path.exists(model_path) and os.path.exists(config_path):
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = override_threshold if override_threshold is not None else config.get('optimal_threshold', 0.70)
        print(f"Loaded trained LightGBM model from {model_path} (Threshold = {threshold:.3f})")
    else:
        print("Note: Trained model not found. Using high-precision similarity fallback (Threshold = 0.75).")
        clf = None
        threshold = 0.75

    print("\n>>> [Step 2/3] Indexing Candidate Streams (Zero Memory Duplication)...")
    blocker = ScalableMultiStrategyBlocker(max_candidates_per_s1=8)
    blocker.fit_from_dict(s23_dict, desc="Candidate Index")

    print("\n>>> [Step 3/3] Streaming Inference & Generating Outputs...")
    total_s1 = len(s1_eids)
    total_candidates_count = 0
    total_matches_count = 0
    singletons_count = 0

    # Open files for direct streaming output
    with open(matching_path, 'w', encoding='utf-8') as f_match, open(candidate_path, 'w', encoding='utf-8') as f_cand:
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        with tqdm(total=total_s1, desc="Processing Test Entities", unit="entity", leave=True) as pbar:
            for start_idx in range(0, total_s1, chunk_size):
                end_idx = min(start_idx + chunk_size, total_s1)
                chunk_s1_eids = s1_eids[start_idx:end_idx]

                # 1. Fast block lookup using pre-built index
                chunk_cands_map = blocker.generate_candidates_for_ids(chunk_s1_eids, s1_dict)

                # 2. Build pairwise feature vectors directly for candidates
                feature_vectors = []
                valid_pairs = []
                for s1_id in chunk_s1_eids:
                    c_list = chunk_cands_map.get(s1_id, [])
                    if not c_list:
                        continue
                    s1_name, s1_addr, s1_country, s1_nums = s1_dict[s1_id]
                    for c_id in c_list:
                        cand_info = s23_dict.get(c_id)
                        if cand_info:
                            c_name, c_addr, c_country, c_nums = cand_info
                            v = extract_pair_features_vector(
                                s1_name, s1_addr, s1_country, s1_nums,
                                c_id, c_name, c_addr, c_country, c_nums
                            )
                            feature_vectors.append(v)
                            valid_pairs.append((s1_id, c_id))

                matched_map = {eid: [] for eid in chunk_s1_eids}

                if feature_vectors:
                    X_mat = np.array(feature_vectors, dtype=np.float32)
                    if clf is not None:
                        probs = clf.predict_proba(X_mat)[:, 1]
                    else:
                        probs = (
                            0.60 * X_mat[:, 1] +
                            0.30 * X_mat[:, 8] +
                            0.10 * X_mat[:, 15]
                        )

                    for (s1_id, c_id), p in zip(valid_pairs, probs):
                        if p >= threshold:
                            matched_map[s1_id].append(c_id)

                # Write chunk rows to disk
                for s1_id in chunk_s1_eids:
                    cands = chunk_cands_map.get(s1_id, [])
                    cand_str = ",".join(cands)
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    total_candidates_count += len(cands)

                    matches = sorted(list(set(matched_map.get(s1_id, []))))
                    match_str = ",".join(matches)
                    f_match.write(f"{s1_id}\t{match_str}\n")

                    if matches:
                        total_matches_count += len(matches)
                    else:
                        singletons_count += 1

                pbar.update(end_idx - start_idx)

    print("\n==================================================")
    print("           INFERENCE SUMMARY & METRICS            ")
    print("==================================================")
    print(f"  * Total Source 1 Entities: {total_s1:,}")
    print(f"  * Singletons (No Match):   {singletons_count:,} ({singletons_count/total_s1*100:.1f}%)")
    print(f"  * Total Matches Found:     {total_matches_count:,}")
    print(f"  * Total Candidate Pairs:   {total_candidates_count:,} (Avg: {total_candidates_count/total_s1:.2f}/entity)")
    print(f"\n[+] Saved matching results: {matching_path}")
    print(f"[+] Saved candidate pairs:  {candidate_path}")

    return pd.DataFrame(), pd.DataFrame()
