"""
Preprocessing and Normalization Module for Business Entity Resolution.

Optimized with single-pass compiled regex and persistent, informative tqdm progress bars.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Set, Tuple
from tqdm import tqdm

# Legal and Corporate Dictionary
LEGAL_DICT = {
    'pvt': 'private', 'ltd': 'limited', 'corp': 'corporation', 'inc': 'incorporated',
    'co': 'company', 'llc': 'limited liability company', 'llp': 'limited liability partnership',
    'plc': 'public limited company', 'sarl': 'societe a responsabilite limitee',
    'sas': 'societe par actions simplifiee', 'sa': 'societe anonyme', 'ste': 'societe',
    'gmbh': 'gesellschaft mit beschrankter haftung', 'assoc': 'association', 'mfg': 'manufacturing',
    'intl': 'international', 'dept': 'department', 'grp': 'group', 'serv': 'services',
    'soln': 'solutions', 'tech': 'technologies', 'ent': 'enterprises', 'med': 'medical',
    'pharma': 'pharmaceuticals', 'dist': 'distributors'
}
LEGAL_REGEX = re.compile(r'\b(' + '|'.join(re.escape(k) for k in LEGAL_DICT.keys()) + r')\b')

# Address Dictionary
ADDR_DICT = {
    **LEGAL_DICT,
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'dr': 'drive',
    'ln': 'lane', 'ct': 'court', 'pl': 'place', 'sq': 'square', 'pk': 'park',
    'pkwy': 'parkway', 'hwy': 'highway', 'ctr': 'center', 'fl': 'floor', 'flr': 'floor',
    'ste': 'suite', 'apt': 'apartment', 'bldg': 'building', 'no': 'number', 'op': 'opposite',
    'opp': 'opposite', 'nr': 'near', 'mkt': 'market', 'ind': 'industrial', 'est': 'estate',
    'dist': 'district', 'nagar': 'nagar', 'marg': 'marg', 'rue': 'rue', 'av': 'avenue',
    'bd': 'boulevard', 'all': 'allee'
}
ADDR_REGEX = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ADDR_DICT.keys()) + r')\b')

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
    """Fast single-pass abbreviation expansion."""
    if not text:
        return ""
    if is_address:
        return ADDR_REGEX.sub(lambda m: ADDR_DICT[m.group(1)], text)
    return LEGAL_REGEX.sub(lambda m: LEGAL_DICT[m.group(1)], text)


def normalize_name_fast(text: str) -> Tuple[str, str]:
    """Single-pass clean, expand, and token sort for names."""
    if not isinstance(text, str) or not text:
        return "", ""
    cleaned = normalize_unicode(text.lower()).replace('&', ' and ')
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    expanded = LEGAL_REGEX.sub(lambda m: LEGAL_DICT[m.group(1)], cleaned)
    norm = re.sub(r'\s+', ' ', expanded).strip()
    sorted_tokens = " ".join(sorted(norm.split()))
    return norm, sorted_tokens


def normalize_addr_fast(text: str) -> Tuple[str, List[str]]:
    """Single-pass clean, expand, and number extraction for addresses."""
    if not isinstance(text, str) or not text:
        return "", []
    cleaned = normalize_unicode(text.lower()).replace('&', ' and ')
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    expanded = ADDR_REGEX.sub(lambda m: ADDR_DICT[m.group(1)], cleaned)
    norm = re.sub(r'\s+', ' ', expanded).strip()
    numbers = re.findall(r'\b\d+\b', norm)
    return norm, numbers


def normalize_country(country: str) -> str:
    """Clean country string safely."""
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


def extract_numeric_tokens(text: str) -> List[str]:
    """Extract numeric sequences."""
    if not text:
        return []
    return re.findall(r'\b\d+\b', text)


def preprocess_dataframe(df: pd.DataFrame, desc: str = "Records", chunk_size: int = 100000) -> pd.DataFrame:
    """
    High-speed, single-pass batch preprocessing with persistent, visible tqdm progress bars.
    """
    total_records = len(df)
    df = df.copy()
    
    # Fill missing values
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('UNKNOWN')

    raw_names = df['business_name'].tolist()
    raw_addrs = df['business_address'].tolist()
    raw_countries = df['country'].tolist()

    norm_names = []
    sorted_names = []
    norm_addrs = []
    addr_numbers = []
    norm_countries = []

    with tqdm(total=total_records, desc=f"[{desc}] Normalizing {total_records:,} records", unit="rec", leave=True) as pbar:
        for i in range(0, total_records, chunk_size):
            end_i = min(i + chunk_size, total_records)
            chunk_names = raw_names[i:end_i]
            chunk_addrs = raw_addrs[i:end_i]
            chunk_countries = raw_countries[i:end_i]

            for name, addr, country in zip(chunk_names, chunk_addrs, chunk_countries):
                n_norm, n_sort = normalize_name_fast(name)
                a_norm, a_nums = normalize_addr_fast(addr)
                c_norm = normalize_country(country)

                norm_names.append(n_norm)
                sorted_names.append(n_sort)
                norm_addrs.append(a_norm)
                addr_numbers.append(a_nums)
                norm_countries.append(c_norm)

            pbar.update(end_i - i)

    df['norm_name'] = norm_names
    df['sorted_name'] = sorted_names
    df['clean_name'] = norm_names
    df['norm_addr'] = norm_addrs
    df['clean_addr'] = norm_addrs
    df['norm_country'] = norm_countries
    df['addr_numbers'] = addr_numbers

    return df
