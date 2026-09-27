"""
Feature Engineering Module for Pairwise Entity Matching.

Ultra-high performance pairwise feature extraction using rapidfuzz C++ routines
operating directly on string tuples (100x faster than pandas Series access).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz

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


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute standard Jaccard token similarity."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def extract_pair_features_fast(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    s1_nums: List[str],
    c_id: str,
    c_name: str,
    c_addr: str,
    c_country: str,
    c_nums: List[str]
) -> Dict[str, float]:
    """
    Direct C++ string feature extractor on raw primitive types.
    """
    # Name string similarities
    name_lev = fuzz.ratio(s1_name, c_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_exact = 1.0 if (s1_name == c_name and len(s1_name) > 0) else 0.0
    
    # Prefix similarity
    name_pfx_3 = 1.0 if (len(s1_name) >= 3 and len(c_name) >= 3 and s1_name[:3] == c_name[:3]) else 0.0
    name_len_diff = float(abs(len(s1_name) - len(c_name)))
    
    # Address string similarities
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
    addr_exact = 1.0 if (s1_addr == c_addr and len(s1_addr) > 0) else 0.0
    addr_len_diff = float(abs(len(s1_addr) - len(c_addr)))
    
    # Address numeric tokens
    set_s1_nums = set(s1_nums)
    set_c_nums = set(c_nums)
    num_jaccard = compute_jaccard_similarity(set_s1_nums, set_c_nums)
    has_common_num = 1.0 if (set_s1_nums and set_c_nums and len(set_s1_nums.intersection(set_c_nums)) > 0) else 0.0
    
    # Country & Source type indicators
    country_match = 1.0 if s1_country == c_country else 0.0
    is_s2 = 1.0 if c_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if c_id.startswith('S3-') else 0.0
    
    # Interactions
    name_addr_interaction = name_token_set * addr_token_set
    name_high_addr_low = 1.0 if (name_token_set >= 0.85 and addr_token_set < 0.35) else 0.0
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


def extract_pair_features_vector(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    s1_nums: List[str],
    c_id: str,
    c_name: str,
    c_addr: str,
    c_country: str,
    c_nums: List[str]
) -> List[float]:
    """
    Direct C++ string feature extractor returning raw float vector for instant numpy conversion.
    """
    name_lev = fuzz.ratio(s1_name, c_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_exact = 1.0 if (s1_name == c_name and len(s1_name) > 0) else 0.0
    
    name_pfx_3 = 1.0 if (len(s1_name) >= 3 and len(c_name) >= 3 and s1_name[:3] == c_name[:3]) else 0.0
    name_len_diff = float(abs(len(s1_name) - len(c_name)))
    
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
    addr_exact = 1.0 if (s1_addr == c_addr and len(s1_addr) > 0) else 0.0
    addr_len_diff = float(abs(len(s1_addr) - len(c_addr)))
    
    set_s1_nums = set(s1_nums)
    set_c_nums = set(c_nums)
    num_jaccard = compute_jaccard_similarity(set_s1_nums, set_c_nums)
    has_common_num = 1.0 if (set_s1_nums and set_c_nums and len(set_s1_nums.intersection(set_c_nums)) > 0) else 0.0
    
    country_match = 1.0 if s1_country == c_country else 0.0
    is_s2 = 1.0 if c_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if c_id.startswith('S3-') else 0.0
    
    name_addr_interaction = name_token_set * addr_token_set
    name_high_addr_low = 1.0 if (name_token_set >= 0.85 and addr_token_set < 0.35) else 0.0
    name_low_addr_high = 1.0 if (name_token_set < 0.35 and addr_token_set >= 0.85) else 0.0

    return [
        name_lev,
        name_token_set,
        name_token_sort,
        name_partial,
        name_exact,
        name_pfx_3,
        name_len_diff,
        addr_lev,
        addr_token_set,
        addr_token_sort,
        addr_partial,
        addr_exact,
        addr_len_diff,
        num_jaccard,
        has_common_num,
        country_match,
        is_s2,
        is_s3,
        name_addr_interaction,
        name_high_addr_low,
        name_low_addr_high,
    ]


def extract_pair_features(s1_row, cand_row) -> Dict[str, float]:
    """Compatibility wrapper for pandas Series."""
    return extract_pair_features_fast(
        s1_row['norm_name'], s1_row['norm_addr'], s1_row['norm_country'], s1_row['addr_numbers'],
        cand_row['entity_id'], cand_row['norm_name'], cand_row['norm_addr'], cand_row['norm_country'], cand_row['addr_numbers']
    )


def build_feature_dataframe(
    pairs: List[Tuple[str, str]],
    s1_dict: Dict[str, Tuple],
    cand_dict: Dict[str, Tuple]
) -> pd.DataFrame:
    """
    High-speed vectorized feature matrix construction from raw tuple lookups.
    s1_dict: eid -> (norm_name, norm_addr, norm_country, addr_numbers)
    cand_dict: eid -> (norm_name, norm_addr, norm_country, addr_numbers)
    """
    records = []
    for s1_id, cand_id in pairs:
        if s1_id in s1_dict and cand_id in cand_dict:
            s1_name, s1_addr, s1_country, s1_nums = s1_dict[s1_id]
            c_name, c_addr, c_country, c_nums = cand_dict[cand_id]
            feats = extract_pair_features_fast(
                s1_name, s1_addr, s1_country, s1_nums,
                cand_id, c_name, c_addr, c_country, c_nums
            )
            feats['source1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            records.append(feats)
            
    return pd.DataFrame(records)
