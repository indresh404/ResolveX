"""
End-to-End Inference Pipeline for Business Entity Resolution.

Loads test datasets, applies country-agnostic preprocessing, runs candidate blocking,
computes pairwise features, applies the trained classifier, and formats the output TSVs
strictly adhering to challenge specifications with real-time tqdm progress bars.
"""

import os
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
    from src.blocking import MultiStrategyBlocker
    from src.features import extract_pair_features, build_feature_dataframe
except ImportError:
    from preprocessing import preprocess_dataframe
    from blocking import MultiStrategyBlocker
    from features import extract_pair_features, build_feature_dataframe


def load_tsv_with_progress(filepath: str, desc: str = "Loading TSV") -> pd.DataFrame:
    """Read large TSV files with a visible progress bar."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    chunksize = 100000
    chunks = []
    with tqdm(desc=f"{desc} ({file_size_mb:.1f} MB)", unit="chunk") as pbar:
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
    batch_size: int = 50000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs full inference and returns (matching_results_df, candidate_pairs_df).
    """
    print("\n>>> [Step 1/4] Loading Test Datasets...")
    s1_raw = load_tsv_with_progress(s1_path, desc="Test Source 1 (Reference)")
    s2_raw = load_tsv_with_progress(s2_path, desc="Test Source 2 (Stream)")
    s3_raw = load_tsv_with_progress(s3_path, desc="Test Source 3 (Stream)")

    print(f"Loaded records: S1 = {len(s1_raw):,} | S2 = {len(s2_raw):,} | S3 = {len(s3_raw):,}")

    print("\n>>> [Step 2/4] Preprocessing & Normalizing Records (Country-Agnostic)...")
    s1_df = preprocess_dataframe(s1_raw, desc="Test S1")
    s2_df = preprocess_dataframe(s2_raw, desc="Test S2")
    s3_df = preprocess_dataframe(s3_raw, desc="Test S3")

    s1_dict = {row['entity_id']: row for _, row in s1_df.iterrows()}
    s23_dict = {row['entity_id']: row for _, row in pd.concat([s2_df, s3_df], ignore_index=True).iterrows()}

    print("\n>>> [Step 3/4] Candidate Generation (Multi-Strategy Blocking)...")
    blocker = MultiStrategyBlocker(top_k_tfidf=25, window_size=10)
    candidate_map = blocker.generate_candidates(s1_df, s2_df, s3_df)

    cand_rows = []
    total_cand_pairs = 0
    inference_pairs = []

    for s1_id in s1_df['entity_id']:
        cands = candidate_map.get(s1_id, [])
        cand_str = ",".join(cands)
        cand_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})
        total_cand_pairs += len(cands)
        for c in cands:
            inference_pairs.append((s1_id, c))

    candidate_pairs_df = pd.DataFrame(cand_rows)
    print(f"Generated {total_cand_pairs:,} candidate pairs across {len(candidate_pairs_df):,} S1 entities.")

    # Load model and config
    if os.path.exists(model_path) and os.path.exists(config_path):
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = override_threshold if override_threshold is not None else config.get('optimal_threshold', 0.70)
        feature_cols = config.get('feature_columns', [])
    else:
        print("Note: Trained model not found. Using fallback similarity thresholding.")
        clf = None
        threshold = 0.75
        feature_cols = []

    print(f"\n>>> [Step 4/4] Scoring Candidate Pairs (Decision Threshold = {threshold:.3f})...")
    matched_map = {s1_id: [] for s1_id in s1_df['entity_id']}

    if inference_pairs:
        num_batches = (len(inference_pairs) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(inference_pairs), batch_size), total=num_batches, desc="Scoring candidate batches"):
            batch_pairs = inference_pairs[i:i + batch_size]
            X_batch_df = build_feature_dataframe(batch_pairs, s1_dict, s23_dict)
            
            if clf is not None and feature_cols:
                X_mat = X_batch_df[feature_cols].values
                probs = clf.predict_proba(X_mat)[:, 1]
            else:
                probs = (
                    0.60 * X_batch_df['name_token_set'].values +
                    0.30 * X_batch_df['addr_token_set'].values +
                    0.10 * X_batch_df['country_match'].values
                )
            
            X_batch_df['prob'] = probs
            high_conf = X_batch_df[X_batch_df['prob'] >= threshold]

            for _, row in high_conf.iterrows():
                matched_map[row['source1_entity_id']].append(row['candidate_entity_id'])

    # Format matching results dataframe
    match_rows = []
    total_matches = 0
    singletons = 0

    for s1_id in s1_df['entity_id']:
        matches = sorted(list(set(matched_map.get(s1_id, []))))
        match_str = ",".join(matches)
        match_rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': match_str})
        if matches:
            total_matches += len(matches)
        else:
            singletons += 1

    matching_results_df = pd.DataFrame(match_rows)
    print(f"\n[+] Inference Summary:")
    print(f"  • Total S1 Entities:     {len(matching_results_df):,}")
    print(f"  • Singletons (no match): {singletons:,} ({singletons / len(matching_results_df) * 100:.1f}%)")
    print(f"  • Total Matches Found:   {total_matches:,}")

    return matching_results_df, candidate_pairs_df
