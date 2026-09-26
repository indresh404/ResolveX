"""
Feature Engineering Module for Pairwise Entity Matching.

Computes fine-grained lexical, phonetic, token, numeric, and cross-field interaction features
between Source 1 and candidate (Source 2 / Source 3) entity pairs using high-performance C++
implementations (rapidfuzz).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz, distance


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute standard Jaccard token similarity."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def extract_pair_features(
    s1_row: pd.Series,
    cand_row: pd.Series
) -> Dict[str, float]:
    """
    Extract comprehensive pairwise feature vector between an S1 entity and a candidate entity.
    """
    # Names
    s1_name = s1_row['norm_name']
    c_name = cand_row['norm_name']
    
    # Addresses
    s1_addr = s1_row['norm_addr']
    c_addr = cand_row['norm_addr']
    
    # Name string similarities (scale to [0, 1])
    name_lev = fuzz.ratio(s1_name, c_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_exact = 1.0 if s1_name == c_name and len(s1_name) > 0 else 0.0
    
    # Prefix similarity
    name_pfx_3 = 1.0 if (s1_name[:3] == c_name[:3] and len(s1_name) >= 3 and len(c_name) >= 3) else 0.0
    name_len_diff = abs(len(s1_name) - len(c_name))
    
    # Address string similarities
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
    addr_exact = 1.0 if s1_addr == c_addr and len(s1_addr) > 0 else 0.0
    addr_len_diff = abs(len(s1_addr) - len(c_addr))
    
    # Address numeric tokens (house numbers, postal codes)
    s1_nums = set(s1_row['addr_numbers'])
    c_nums = set(cand_row['addr_numbers'])
    num_jaccard = compute_jaccard_similarity(s1_nums, c_nums)
    has_common_num = 1.0 if (s1_nums and c_nums and len(s1_nums.intersection(c_nums)) > 0) else 0.0
    
    # Country match
    country_match = 1.0 if s1_row['norm_country'] == cand_row['norm_country'] else 0.0
    
    # Source type indicator
    cand_id = cand_row['entity_id']
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0
    
    # Cross-field interactions
    name_addr_interaction = name_token_set * addr_token_set
    # Chain/Franchise indicator: same name, different address
    name_high_addr_low = 1.0 if (name_token_set >= 0.85 and addr_token_set < 0.35) else 0.0
    # Same building indicator: different business name, matching address
    name_low_addr_high = 1.0 if (name_token_set < 0.35 and addr_token_set >= 0.85) else 0.0

    return {
        'name_lev': name_lev,
        'name_token_set': name_token_set,
        'name_token_sort': name_token_sort,
        'name_partial': name_partial,
        'name_exact': name_exact,
        'name_pfx_3': name_pfx_3,
        'name_len_diff': name_len_diff,
        'addr_lev': addr_lev,
        'addr_token_set': addr_token_set,
        'addr_token_sort': addr_token_sort,
        'addr_partial': addr_partial,
        'addr_exact': addr_exact,
        'addr_len_diff': addr_len_diff,
        'num_jaccard': num_jaccard,
        'has_common_num': has_common_num,
        'country_match': country_match,
        'is_s2': is_s2,
        'is_s3': is_s3,
        'name_addr_interaction': name_addr_interaction,
        'name_high_addr_low': name_high_addr_low,
        'name_low_addr_high': name_low_addr_high,
    }


def build_feature_dataframe(
    pairs: List[Tuple[str, str]],
    s1_dict: Dict[str, pd.Series],
    cand_dict: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Build feature matrix for a list of (s1_id, candidate_id) pairs.
    """
    records = []
    for s1_id, cand_id in pairs:
        if s1_id in s1_dict and cand_id in cand_dict:
            feats = extract_pair_features(s1_dict[s1_id], cand_dict[cand_id])
            feats['source1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            records.append(feats)
            
    return pd.DataFrame(records)
