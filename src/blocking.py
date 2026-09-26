"""
Blocking (Candidate Generation) Module for Business Entity Resolution.

Implements multi-strategy candidate generation:
1. TF-IDF Character n-gram Cosine Similarity
2. Inverted Index Token Overlap (Jaccard)
3. Sorted Neighborhood Prefix Windowing
4. Country-partitioned candidate filtering

Goal: High Pair Completeness (Recall >= 0.98) with High Reduction Ratio.
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class MultiStrategyBlocker:
    def __init__(
        self,
        top_k_tfidf: int = 25,
        tfidf_ngram_range: Tuple[int, int] = (2, 4),
        window_size: int = 10,
        min_token_len: int = 3,
        max_token_freq_ratio: float = 0.05,
    ):
        self.top_k_tfidf = top_k_tfidf
        self.tfidf_ngram_range = tfidf_ngram_range
        self.window_size = window_size
        self.min_token_len = min_token_len
        self.max_token_freq_ratio = max_token_freq_ratio

    def _build_inverted_index(self, s23_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Build token -> list of S2/S3 entity_ids index."""
        token_to_ids = defaultdict(list)
        total_docs = len(s23_df)
        max_freq = max(5, int(total_docs * self.max_token_freq_ratio))

        # First pass: count token frequencies
        token_counts = defaultdict(int)
        for _, row in s23_df.iterrows():
            tokens = set(row['norm_name'].split())
            for t in tokens:
                if len(t) >= self.min_token_len:
                    token_counts[t] += 1

        # Second pass: index only non-frequent significant tokens
        for _, row in s23_df.iterrows():
            eid = row['entity_id']
            tokens = set(row['norm_name'].split())
            for t in tokens:
                if len(t) >= self.min_token_len and token_counts[t] <= max_freq:
                    token_to_ids[t].append(eid)

        return token_to_ids

    def _get_token_candidates(
        self, s1_df: pd.DataFrame, token_to_ids: Dict[str, List[str]]
    ) -> Dict[str, Set[str]]:
        """Find candidate matches for each S1 entity using token overlap."""
        candidates = defaultdict(set)
        for _, row in s1_df.iterrows():
            s1_id = row['entity_id']
            tokens = set(row['norm_name'].split())
            for t in tokens:
                if t in token_to_ids:
                    for s23_id in token_to_ids[t]:
                        candidates[s1_id].add(s23_id)
        return candidates

    def _get_tfidf_candidates(
        self, s1_df: pd.DataFrame, s23_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """Generate candidates using character n-gram TF-IDF Nearest Neighbors."""
        candidates = defaultdict(set)
        if s1_df.empty or s23_df.empty:
            return candidates

        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=self.tfidf_ngram_range,
            min_df=1,
            sublinear_tf=True
        )

        all_names = list(s23_df['norm_name'].values) + list(s1_df['norm_name'].values)
        vectorizer.fit(all_names)

        s23_matrix = vectorizer.transform(s23_df['norm_name'].values)
        s1_matrix = vectorizer.transform(s1_df['norm_name'].values)

        k = min(self.top_k_tfidf, s23_matrix.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(s23_matrix)

        distances, indices = nn.kneighbors(s1_matrix)

        s1_ids = s1_df['entity_id'].values
        s23_ids = s23_df['entity_id'].values

        for i, s1_id in enumerate(s1_ids):
            for neighbor_idx in indices[i]:
                candidates[s1_id].add(s23_ids[neighbor_idx])

        return candidates

    def _get_sorted_neighborhood_candidates(
        self, s1_df: pd.DataFrame, s23_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """Generate candidates via sliding window on alphabetically sorted normalized names."""
        candidates = defaultdict(set)
        
        combined = []
        for _, row in s1_df.iterrows():
            combined.append((row['norm_name'], 'S1', row['entity_id']))
        for _, row in s23_df.iterrows():
            combined.append((row['norm_name'], 'S23', row['entity_id']))

        # Sort combined records by normalized name
        combined.sort(key=lambda x: x[0])

        w = self.window_size
        n = len(combined)

        for i in range(n):
            name_i, src_i, id_i = combined[i]
            if src_i != 'S1':
                continue

            # Slide window around index i
            start_idx = max(0, i - w)
            end_idx = min(n, i + w + 1)

            for j in range(start_idx, end_idx):
                if j == i:
                    continue
                name_j, src_j, id_j = combined[j]
                if src_j == 'S23':
                    # Check if prefix matches or similarity is reasonable
                    if name_i[:3] == name_j[:3] or abs(len(name_i) - len(name_j)) <= 3:
                        candidates[id_i].add(id_j)

        return candidates

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Run multi-strategy candidate generation partitioned by normalized country.
        Returns mapping: s1_entity_id -> list of candidate S2/S3 entity_ids.
        """
        # Combine S2 and S3
        s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

        final_candidates = defaultdict(set)

        # Partition by country to speed up and improve precision
        countries = set(s1_df['norm_country'].unique()).union(set(s23_df['norm_country'].unique()))

        for country in countries:
            sub_s1 = s1_df[s1_df['norm_country'] == country]
            sub_s23 = s23_df[s23_df['norm_country'] == country]

            if sub_s1.empty or sub_s23.empty:
                # Handle cases with no country match or unknown
                sub_s23 = s23_df

            # Strategy 1: TF-IDF
            tfidf_cands = self._get_tfidf_candidates(sub_s1, sub_s23)

            # Strategy 2: Token Inverted Index
            token_index = self._build_inverted_index(sub_s23)
            token_cands = self._get_token_candidates(sub_s1, token_index)

            # Strategy 3: Sorted Neighborhood
            sn_cands = self._get_sorted_neighborhood_candidates(sub_s1, sub_s23)

            # Union candidates per S1 entity
            for _, row in sub_s1.iterrows():
                s1_id = row['entity_id']
                unioned = tfidf_cands[s1_id].union(token_cands[s1_id]).union(sn_cands[s1_id])
                final_candidates[s1_id].update(unioned)

        # Ensure every S1 record has an entry in output (even if empty)
        result = {}
        for s1_id in s1_df['entity_id']:
            result[s1_id] = sorted(list(final_candidates[s1_id]))

        return result
