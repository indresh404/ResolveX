"""
Training Module for Business Entity Matching.

Trains a precision-optimized LightGBM classifier on engineered pairwise features
with entity-level validation splitting, balanced negative sampling, threshold optimization
directly targeting Macro F_0.5, and real-time tqdm progress bars throughout.
"""

import os
import sys
import json
import pickle
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
    from src.evaluate import compute_entity_f05, evaluate_predictions, evaluate_blocking
except ImportError:
    from preprocessing import preprocess_dataframe
    from blocking import MultiStrategyBlocker
    from features import extract_pair_features, build_feature_dataframe
    from evaluate import compute_entity_f05, evaluate_predictions, evaluate_blocking


FEATURE_COLUMNS = [
    'name_lev',
    'name_token_set',
    'name_token_sort',
    'name_partial',
    'name_exact',
    'name_pfx_3',
    'name_len_diff',
    'addr_lev',
    'addr_token_set',
    'addr_token_sort',
    'addr_partial',
    'addr_exact',
    'addr_len_diff',
    'num_jaccard',
    'has_common_num',
    'country_match',
    'is_s2',
    'is_s3',
    'name_addr_interaction',
    'name_high_addr_low',
    'name_low_addr_high',
]


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


def train_matching_model(
    train_s1_path: str = 'student_resource/dataset/train/train_source1.tsv',
    train_s2_path: str = 'student_resource/dataset/train/train_source2.tsv',
    train_s3_path: str = 'student_resource/dataset/train/train_source3.tsv',
    train_gt_path: str = 'student_resource/dataset/train/train_ground_truth.tsv',
    model_output_dir: str = 'models',
    val_size: float = 0.2,
    random_state: int = 42,
    sample_limit: int = None
):
    os.makedirs(model_output_dir, exist_ok=True)
    random.seed(random_state)
    np.random.seed(random_state)

    print("\n=======================================================")
    print("      RESOLVEX: MODEL TRAINING & THRESHOLD TUNING      ")
    print("=======================================================")

    print("\n[Step 1/5] Loading raw datasets...")
    s1_raw = load_tsv_with_progress(train_s1_path, desc="Loading Source 1 (Reference)")
    s2_raw = load_tsv_with_progress(train_s2_path, desc="Loading Source 2 (Stream)")
    s3_raw = load_tsv_with_progress(train_s3_path, desc="Loading Source 3 (Stream)")
    gt_df = load_tsv_with_progress(train_gt_path, desc="Loading Ground Truth")

    if sample_limit is not None and len(s1_raw) > sample_limit:
        print(f"Subsampling S1 reference to {sample_limit} records for rapid development...")
        s1_raw = s1_raw.head(sample_limit)
        gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_raw['entity_id'])]

    print("\n[Step 2/5] Preprocessing text & normalizing records...")
    s1_df = preprocess_dataframe(s1_raw, desc="Source 1")
    s2_df = preprocess_dataframe(s2_raw, desc="Source 2")
    s3_df = preprocess_dataframe(s3_raw, desc="Source 3")

    s1_dict = {row['entity_id']: row for _, row in s1_df.iterrows()}
    s23_dict = {row['entity_id']: row for _, row in pd.concat([s2_df, s3_df], ignore_index=True).iterrows()}

    # Ground truth mapping
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        val = row['matched_entity_ids']
        gt_map[s1_id] = set([x.strip() for x in str(val).split(',') if x.strip()]) if pd.notna(val) else set()

    # Entity-level Train/Validation Split (prevents leakage)
    all_s1_ids = list(s1_df['entity_id'].values)
    train_s1_ids, val_s1_ids = train_test_split(all_s1_ids, test_size=val_size, random_state=random_state)
    print(f"\nEntity Split: Train = {len(train_s1_ids):,} entities | Validation = {len(val_s1_ids):,} entities")

    train_s1_df = s1_df[s1_df['entity_id'].isin(train_s1_ids)].copy()
    val_s1_df = s1_df[s1_df['entity_id'].isin(val_s1_ids)].copy()

    print("\n[Step 3/5] Candidate Generation (Stage 1 Multi-Strategy Blocking)...")
    blocker = MultiStrategyBlocker(top_k_tfidf=25, window_size=10)
    
    print("Generating training candidate blocks...")
    train_candidates = blocker.generate_candidates(train_s1_df, s2_df, s3_df)
    print("Generating validation candidate blocks...")
    val_candidates = blocker.generate_candidates(val_s1_df, s2_df, s3_df)

    # Evaluate blocking quality on validation
    val_gt_sub = gt_df[gt_df['source1_entity_id'].isin(val_s1_ids)]
    blocking_stats = evaluate_blocking(val_gt_sub, val_candidates, len(s23_dict))
    print(f"\n>>> Validation Blocking Quality:")
    print(f"  • Pair Completeness (Recall Ceiling): {blocking_stats['pair_completeness'] * 100:.2f}%")
    print(f"  • Reduction Ratio:                    {blocking_stats['reduction_ratio'] * 100:.4f}%")
    print(f"  • Avg Candidates per S1 Entity:       {blocking_stats['avg_candidates_per_s1']:.1f}")

    print("\n[Step 4/5] Building Training Pairs & Extracting Features...")
    train_pairs = []
    train_labels = []

    for s1_id in tqdm(train_s1_ids, desc="Sampling training pairs"):
        true_set = gt_map.get(s1_id, set())
        cand_list = train_candidates.get(s1_id, [])

        # 1. Positives (all ground truth matches)
        for true_id in true_set:
            if true_id in s23_dict:
                train_pairs.append((s1_id, true_id))
                train_labels.append(1)

        # 2. Hard Negatives (candidates surviving blocking but not true matches)
        hard_negs = [c for c in cand_list if c not in true_set]
        sample_k = min(len(hard_negs), 10)
        for neg_id in random.sample(hard_negs, sample_k):
            train_pairs.append((s1_id, neg_id))
            train_labels.append(0)

    print(f"Generated {len(train_pairs):,} training pairs (Pos: {sum(train_labels):,}, Neg: {len(train_labels) - sum(train_labels):,})")

    # Feature extraction with progress bar
    print("Extracting features for training pairs...")
    X_train_df = build_feature_dataframe(train_pairs, s1_dict, s23_dict)
    X_train = X_train_df[FEATURE_COLUMNS].values
    y_train = np.array(train_labels)

    # Validation pairs feature extraction
    print("Extracting features for validation candidates...")
    val_pairs = []
    for s1_id in val_s1_ids:
        for cand_id in val_candidates.get(s1_id, []):
            val_pairs.append((s1_id, cand_id))

    X_val_df = build_feature_dataframe(val_pairs, s1_dict, s23_dict)
    X_val = X_val_df[FEATURE_COLUMNS].values if not X_val_df.empty else np.zeros((0, len(FEATURE_COLUMNS)))

    # Train LightGBM Classifier
    print("\n[Step 5/5] Training LightGBM Model & Sweeping F_0.5 Threshold...")
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(pos_weight, 5.0),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)

    # Score validation candidates
    val_preds_prob = clf.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    X_val_df['match_prob'] = val_preds_prob

    # Optimize Decision Threshold directly for Macro F_0.5
    best_threshold = 0.70
    best_f05 = -1.0
    best_stats = {}

    threshold_grid = np.linspace(0.40, 0.95, 23)
    for thresh in tqdm(threshold_grid, desc="Sweeping thresholds"):
        pred_map = {s1_id: set() for s1_id in val_s1_ids}
        above_thresh = X_val_df[X_val_df['match_prob'] >= thresh]
        for _, row in above_thresh.iterrows():
            pred_map[row['source1_entity_id']].add(row['candidate_entity_id'])

        pred_rows = [{'source1_entity_id': k, 'matched_entity_ids': ",".join(sorted(v))} for k, v in pred_map.items()]
        pred_df = pd.DataFrame(pred_rows)

        eval_res = evaluate_predictions(val_gt_sub, pred_df)
        if eval_res['macro_f05'] > best_f05:
            best_f05 = eval_res['macro_f05']
            best_threshold = float(thresh)
            best_stats = eval_res

    print("\n=======================================================")
    print("                TRAINING & EVALUATION RESULTS          ")
    print("=======================================================")
    print(f"  • Optimal Decision Threshold (θ*): {best_threshold:.3f}")
    print(f"  • Validation Macro F_0.5:          {best_stats.get('macro_f05', 0.0):.4f}")
    print(f"  • Validation Precision:            {best_stats.get('macro_precision', 0.0):.4f}")
    print(f"  • Validation Recall:               {best_stats.get('macro_recall', 0.0):.4f}")

    # Save artifacts
    model_path = os.path.join(model_output_dir, 'lgbm_matcher.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    config = {
        'optimal_threshold': best_threshold,
        'feature_columns': FEATURE_COLUMNS,
        'val_macro_f05': best_stats.get('macro_f05', 0.0),
        'val_precision': best_stats.get('macro_precision', 0.0),
        'val_recall': best_stats.get('macro_recall', 0.0),
        'blocking_recall': blocking_stats['pair_completeness'],
        'reduction_ratio': blocking_stats['reduction_ratio']
    }
    config_path = os.path.join(model_output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[+] Saved model:  {model_path}")
    print(f"[+] Saved config: {config_path}")
    return config


if __name__ == '__main__':
    train_matching_model()
