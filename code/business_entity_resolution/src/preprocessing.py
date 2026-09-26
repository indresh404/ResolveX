"""
Preprocessing and Normalization Module for Business Entity Resolution.

Handles string normalization, country-agnostic abbreviation expansions (legal & address),
canonical token set extraction, and numeric token parsing with real-time tqdm progress bars.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Set, Tuple
from tqdm import tqdm

# Common legal and corporate abbreviations across US, India, and EU (France, etc.)
LEGAL_ABBREVIATIONS = {
    r'\bpvt\b': 'private',
    r'\bltd\b': 'limited',
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bco\b': 'company',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\bplc\b': 'public limited company',
    r'\bsarl\b': 'societe a responsabilite limitee',
    r'\bsas\b': 'societe par actions simplifiee',
    r'\bsa\b': 'societe anonyme',
    r'\bste\b': 'societe',
    r'\bgmbh\b': 'gesellschaft mit beschrankter haftung',
    r'\bassoc\b': 'association',
    r'\bmfg\b': 'manufacturing',
    r'\bintl\b': 'international',
    r'\bdept\b': 'department',
    r'\bgrp\b': 'group',
    r'\bserv\b': 'services',
    r'\bsoln\b': 'solutions',
    r'\btech\b': 'technologies',
    r'\bent\b': 'enterprises',
    r'\bmed\b': 'medical',
    r'\bpharma\b': 'pharmaceuticals',
    r'\bdist\b': 'distributors',
}

# Common address abbreviations
ADDRESS_ABBREVIATIONS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bpk\b': 'park',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bctr\b': 'center',
    r'\bfl\b': 'floor',
    r'\bflr\b': 'floor',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bbldg\b': 'building',
    r'\bno\b': 'number',
    r'\bop\b': 'opposite',
    r'\bopp\b': 'opposite',
    r'\bnr\b': 'near',
    r'\bmkt\b': 'market',
    r'\bind\b': 'industrial',
    r'\best\b': 'estate',
    r'\bdist\b': 'district',
    r'\bnagar\b': 'nagar',
    r'\bmarg\b': 'marg',
    r'\brue\b': 'rue',
    r'\bav\b': 'avenue',
    r'\bbd\b': 'boulevard',
    r'\ball\b': 'allee',
}

STOP_WORDS = {
    'the', 'and', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an', 'by',
    'de', 'la', 'le', 'et', 'du', 'des', 'en', 'les'
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (e.g., accented French characters to ASCII)."""
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_text_basic(text: str) -> str:
    """Basic lowercasing, symbol replacement, and punctuation cleanup."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalize_unicode(text.lower())
    text = text.replace('&', ' and ')
    text = text.replace('/', ' ')
    text = text.replace('-', ' ')
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.replace('#', ' ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def expand_abbreviations(text: str, is_address: bool = False) -> str:
    """Expand standard legal and address abbreviations in a country-agnostic manner."""
    if not text:
        return ""
    
    for pattern, replacement in LEGAL_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
        
    if is_address:
        for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text)
            
    return re.sub(r'\s+', ' ', text).strip()


def get_canonical_tokens(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize and return sorted list of unique tokens."""
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return sorted(list(set(tokens)))


def get_sorted_token_str(text: str) -> str:
    """Return space-separated string of alphabetically sorted tokens."""
    tokens = text.split()
    return " ".join(sorted(tokens))


def extract_numeric_tokens(text: str) -> List[str]:
    """Extract numeric sequences (PIN codes, house numbers, postal codes)."""
    if not text:
        return []
    return re.findall(r'\b\d+\b', text)


def normalize_country(country: str) -> str:
    """Clean country string safely without hardcoded assumptions."""
    if not isinstance(country, str) or not country.strip():
        return "UNKNOWN"
    country_clean = country.strip().upper()
    if country_clean in ['INDIA', 'IN', 'IND']:
        return 'INDIA'
    if country_clean in ['US', 'USA', 'UNITED STATES', 'UNITED STATES OF AMERICA']:
        return 'US'
    if country_clean in ['FR', 'FRA', 'FRANCE']:
        return 'FRANCE'
    return country_clean


def preprocess_dataframe(df: pd.DataFrame, desc: str = "Records") -> pd.DataFrame:
    """
    Applies complete preprocessing pipeline to DataFrame with real-time tqdm progress bars.
    """
    df = df.copy()
    
    # Fill missing values
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('UNKNOWN')
    
    # Clean and expand names
    raw_names = df['business_name'].tolist()
    clean_names = [clean_text_basic(x) for x in tqdm(raw_names, desc=f"[{desc}] 1/4 Cleaning names", leave=False)]
    norm_names = [expand_abbreviations(x, is_address=False) for x in tqdm(clean_names, desc=f"[{desc}] 2/4 Expanding names", leave=False)]
    sorted_names = [get_sorted_token_str(x) for x in norm_names]
    
    # Clean and expand addresses
    raw_addrs = df['business_address'].tolist()
    clean_addrs = [clean_text_basic(x) for x in tqdm(raw_addrs, desc=f"[{desc}] 3/4 Cleaning addresses", leave=False)]
    norm_addrs = [expand_abbreviations(x, is_address=True) for x in tqdm(clean_addrs, desc=f"[{desc}] 4/4 Expanding addresses", leave=False)]
    
    # Normalized country & address numbers
    df['clean_name'] = clean_names
    df['norm_name'] = norm_names
    df['sorted_name'] = sorted_names
    df['clean_addr'] = clean_addrs
    df['norm_addr'] = norm_addrs
    df['norm_country'] = [normalize_country(x) for x in df['country'].tolist()]
    df['addr_numbers'] = [extract_numeric_tokens(x) for x in clean_addrs]
    
    return df
