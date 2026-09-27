"""
Scalable Multi-Strategy Candidate Blocking for Amazon-Scale Business Entity Resolution.

High-Speed Inverted Index with One-Time Indexing and Fast Vectorized Queries:
- Indexing 10M S2/S3 records is performed ONCE in memory with minimal memory footprint
- Direct hash lookups with frequency capping (prevents high RAM explosion)
- Ultra-low RAM dictionary referencing and aggressive garbage collection
"""

import gc
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz


class ScalableMultiStrategyBlocker:
    """
    Amazon-scale entity resolution blocker.
    Linear O(N) indexing with hit accumulation and frequency-capped inverted lists.
    Optimized for low-RAM streaming footprint.
    """
    def __init__(
        self,
        max_candidates_per_s1: int = 8,
        min_token_len: int = 3,
        max_key_freq: int = 8000,
        min_similarity_floor: float = 0.30
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_token_len = min_token_len
        self.max_key_freq = max_key_freq
        self.min_similarity_floor = min_similarity_floor
        
        self.is_indexed = False
        self.s23_dict = None
        self.inverted_index = {}
        self.num_index = {}

    def fit_from_dict(self, s23_dict: Dict[str, Tuple], desc: str = "Candidate Index"):
        """
        Builds the inverted index directly from the shared s23_dict without memory duplication.
        s23_dict: eid -> (norm_name, norm_addr, norm_country, addr_numbers)
        """
        print(f"[{desc}] Building index across {len(s23_dict):,} candidate stream records...")
        self.s23_dict = s23_dict

        inv_idx = defaultdict(list)
        num_idx = defaultdict(list)
        min_tok_len = self.min_token_len

        for eid, (name, addr, country, nums) in s23_dict.items():
            if len(name) >= 4:
                inv_idx[(country, 'pfx4', name[:4])].append(eid)
            elif len(name) >= 3:
                inv_idx[(country, 'pfx3', name[:3])].append(eid)

            # Up to 4 significant tokens per entity
            tokens = set(name.split())
            tok_count = 0
            for t in tokens:
                if len(t) >= min_tok_len:
                    inv_idx[(country, 'tok', t)].append(eid)
                    tok_count += 1
                    if tok_count >= 4:
                        break

            # Numeric address tokens (up to 2 numbers)
            for num in nums[:2]:
                if len(num) >= 2:
                    num_idx[(country, num)].append(eid)

        # Prune high-frequency uninformative keys to keep memory minimal
        max_freq = self.max_key_freq
        self.inverted_index = {k: v for k, v in inv_idx.items() if len(v) <= max_freq}
        self.num_index = {k: v for k, v in num_idx.items() if len(v) <= max_freq}
        del inv_idx, num_idx
        gc.collect()

        self.is_indexed = True
        print(f"[{desc}] Index built successfully ({len(self.inverted_index):,} keys)!")

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame, desc: str = "Stream Indexing"):
        """
        Builds index from DataFrames (backward compatible).
        """
        s23_dict = {}
        for eid, name, addr, country, nums in zip(
            s2_df['entity_id'], s2_df['norm_name'], s2_df['norm_addr'], s2_df['norm_country'], s2_df['addr_numbers']
        ):
            s23_dict[eid] = (name, addr, country, tuple(nums))
        for eid, name, addr, country, nums in zip(
            s3_df['entity_id'], s3_df['norm_name'], s3_df['norm_addr'], s3_df['norm_country'], s3_df['addr_numbers']
        ):
            s23_dict[eid] = (name, addr, country, tuple(nums))
        self.fit_from_dict(s23_dict, desc=desc)

    def generate_candidates_for_ids(
        self,
        s1_eids: List[str],
        s1_dict: Dict[str, Tuple]
    ) -> Dict[str, List[str]]:
        """
        Fast candidate retrieval for a list of S1 entity IDs using s1_dict.
        """
        result_candidates = {}
        inv_get = self.inverted_index.get
        num_get = self.num_index.get
        s23_dict = self.s23_dict
        min_tok_len = self.min_token_len
        min_floor = self.min_similarity_floor
        max_cands = self.max_candidates_per_s1

        for s1_id in s1_eids:
            s1_info = s1_dict.get(s1_id)
            if not s1_info:
                result_candidates[s1_id] = []
                continue
            s1_name, s1_addr, country, s1_nums = s1_info

            candidate_hits = {}

            # 1. Prefix key lookups
            if len(s1_name) >= 4:
                pfx4_list = inv_get((country, 'pfx4', s1_name[:4]))
                if pfx4_list:
                    for cid in pfx4_list:
                        candidate_hits[cid] = candidate_hits.get(cid, 0) + 2
            elif len(s1_name) >= 3:
                pfx3_list = inv_get((country, 'pfx3', s1_name[:3]))
                if pfx3_list:
                    for cid in pfx3_list:
                        candidate_hits[cid] = candidate_hits.get(cid, 0) + 2

            # 2. Token lookups
            tok_count = 0
            for t in set(s1_name.split()):
                if len(t) >= min_tok_len:
                    t_list = inv_get((country, 'tok', t))
                    if t_list:
                        for cid in t_list:
                            candidate_hits[cid] = candidate_hits.get(cid, 0) + 1
                    tok_count += 1
                    if tok_count >= 4:
                        break

            # 3. Numeric address lookup
            for num in s1_nums[:2]:
                if len(num) >= 2:
                    n_list = num_get((country, num))
                    if n_list:
                        for cid in n_list:
                            candidate_hits[cid] = candidate_hits.get(cid, 0) + 1

            if not candidate_hits:
                result_candidates[s1_id] = []
                continue

            # Pick top candidates with highest shared hits
            if len(candidate_hits) <= 12:
                top_hit_candidates = list(candidate_hits.keys())
            else:
                top_hit_candidates = sorted(candidate_hits, key=candidate_hits.get, reverse=True)[:12]

            # Fast fuzzy refinement
            scored_candidates = []
            for cand_id in top_hit_candidates:
                cand_info = s23_dict.get(cand_id)
                if not cand_info:
                    continue
                c_name, c_addr, _, _ = cand_info
                name_sim = fuzz.token_set_ratio(s1_name, c_name) / 100.0
                if name_sim < 0.25:
                    continue
                addr_sim = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
                combined_score = 0.65 * name_sim + 0.35 * addr_sim

                if combined_score >= min_floor or name_sim >= 0.70:
                    scored_candidates.append((cand_id, combined_score))

            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                result_candidates[s1_id] = [cid for cid, score in scored_candidates[:max_cands]]
            else:
                result_candidates[s1_id] = []

        return result_candidates

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame = None,
        s3_df: pd.DataFrame = None,
        desc: str = "Blocking"
    ) -> Dict[str, List[str]]:
        if not self.is_indexed:
            if s2_df is None or s3_df is None:
                raise ValueError("Must provide s2_df and s3_df on first run to index streams.")
            self.fit(s2_df, s3_df, desc="Initial Indexing")
        s1_dict = {
            eid: (name, addr, country, nums)
            for eid, name, addr, country, nums in zip(
                s1_df['entity_id'], s1_df['norm_name'], s1_df['norm_addr'], s1_df['norm_country'], s1_df['addr_numbers']
            )
        }
        return self.generate_candidates_for_ids(s1_df['entity_id'].tolist(), s1_dict)


# Backward compatibility alias
MultiStrategyBlocker = ScalableMultiStrategyBlocker
