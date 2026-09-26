"""
Evaluation Module for Business Entity Resolution.

Implements the exact macro-averaged F_0.5 metric, singleton scoring rules,
blocking pair completeness (recall ceiling), and reduction ratio.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str]
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_0.5 for a single S1 entity.
    
    Special Cases:
    - Singleton True (no true matches):
      - If pred is empty -> F_0.5 = 1.0, Precision = 1.0, Recall = 1.0
      - If pred is non-empty -> F_0.5 = 0.0, Precision = 0.0, Recall = 0.0
    - Non-Singleton True (has true matches):
      - If pred is empty -> F_0.5 = 0.0, Precision = 0.0, Recall = 0.0
      - If pred is non-empty -> standard formula
    """
    if not true_matches:
        if not pred_matches:
            return 1.0, 1.0, 1.0  # Perfect singleton prediction
        else:
            return 0.0, 0.0, 0.0  # False merge penalty

    if not pred_matches:
        return 0.0, 0.0, 0.0  # Missed all matches

    true_positives = len(true_matches.intersection(pred_matches))
    precision = float(true_positives) / float(len(pred_matches))
    recall = float(true_positives) / float(len(true_matches))

    if precision + recall == 0:
        return 0.0, precision, recall

    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    numerator = 1.25 * precision * recall
    denominator = 0.25 * precision + recall
    f05 = numerator / denominator if denominator > 0 else 0.0

    return f05, precision, recall


def evaluate_predictions(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5, Precision, and Recall across all S1 entities.
    """
    # Parse ground truth
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        val = row['matched_entity_ids']
        if pd.isna(val) or not str(val).strip():
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set([x.strip() for x in str(val).split(',') if x.strip()])

    # Parse predictions
    pred_map = {}
    for _, row in predictions_df.iterrows():
        s1_id = row['source1_entity_id']
        val = row['matched_entity_ids']
        if pd.isna(val) or not str(val).strip():
            pred_map[s1_id] = set()
        else:
            pred_map[s1_id] = set([x.strip() for x in str(val).split(',') if x.strip()])

    f05_scores = []
    prec_scores = []
    rec_scores = []

    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        f05, prec, rec = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f05)
        prec_scores.append(prec)
        rec_scores.append(rec)

    return {
        'macro_f05': float(np.mean(f05_scores)),
        'macro_precision': float(np.mean(prec_scores)),
        'macro_recall': float(np.mean(rec_scores)),
        'total_entities': len(f05_scores)
    }


def evaluate_blocking(
    ground_truth_df: pd.DataFrame,
    candidate_pairs_map: Dict[str, List[str]],
    total_s23_count: int
) -> Dict[str, float]:
    """
    Evaluate blocking quality:
    - Pair Completeness (Recall Ceiling): fraction of true positive pairs found in candidates
    - Reduction Ratio: 1 - (candidates / total possible pairs)
    """
    total_true_pairs = 0
    found_true_pairs = 0
    total_candidates = 0
    total_s1 = len(ground_truth_df)

    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        val = row['matched_entity_ids']
        true_set = set([x.strip() for x in str(val).split(',') if x.strip()]) if pd.notna(val) else set()
        
        cand_set = set(candidate_pairs_map.get(s1_id, []))
        total_candidates += len(cand_set)
        
        total_true_pairs += len(true_set)
        found_true_pairs += len(true_set.intersection(cand_set))

    pair_completeness = float(found_true_pairs) / float(total_true_pairs) if total_true_pairs > 0 else 1.0
    total_possible_pairs = total_s1 * total_s23_count
    reduction_ratio = 1.0 - (float(total_candidates) / float(total_possible_pairs)) if total_possible_pairs > 0 else 1.0

    return {
        'pair_completeness': pair_completeness,
        'reduction_ratio': reduction_ratio,
        'found_true_pairs': found_true_pairs,
        'total_true_pairs': total_true_pairs,
        'total_candidates': total_candidates,
        'avg_candidates_per_s1': float(total_candidates) / float(total_s1) if total_s1 > 0 else 0.0
    }
