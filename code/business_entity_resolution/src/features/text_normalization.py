import re
import unicodedata
import pandas as pd
from urllib.parse import urlparse

# Standard corporate suffixes to strip or normalize
LEGAL_SUFFIXES = {
    r'\binc\b\.?': '',
    r'\bllc\b\.?': '',
    r'\bl\.l\.c\.\b': '',
    r'\bltd\b\.?': '',
    r'\blimited\b': '',
    r'\bcorp\b\.?': '',
    r'\bcorporation\b': '',
    r'\bco\b\.?': '',
    r'\bcompany\b': '',
    r'\bplc\b\.?': '',
    r'\bgmbh\b': '',
    r'\bsa\b': '',
    r'\bnv\b': '',
    r'\bbv\b': '',
    r'\bsrl\b': '',
    r'\bspa\b': '',
}

# Common address abbreviations
ADDRESS_ABBREV = {
    r'\bst\b\.?': 'street',
    r'\bave\b\.?': 'avenue',
    r'\brd\b\.?': 'road',
    r'\bblvd\b\.?': 'boulevard',
    r'\bdr\b\.?': 'drive',
    r'\bln\b\.?': 'lane',
    r'\bct\b\.?': 'court',
    r'\bpl\b\.?': 'place',
    r'\bsq\b\.?': 'square',
    r'\bste\b\.?': 'suite',
    r'\bapt\b\.?': 'apartment',
    r'\bpkwy\b\.?': 'parkway',
    r'\bhwy\b\.?': 'highway',
}

# Business abbreviations
BUSINESS_ABBREV = {
    r'\bmfg\b\.?': 'manufacturing',
    r'\bmgmt\b\.?': 'management',
    r'\bintl\b\.?': 'international',
    r'\bgrp\b\.?': 'group',
    r'\bassoc\b\.?': 'associates',
    r'\btech\b\.?': 'technology',
    r'\bbros\b\.?': 'brothers',
    r'\bctr\b\.?': 'center',
}

# Plural stemming map
STEMMING_MAP = {
    r'\bservices\b': 'service',
    r'\bpartners\b': 'partner',
    r'\bholdings\b': 'holding',
    r'\bstores\b': 'store',
    r'\bcenters\b': 'center',
}

STOPWORDS = {'the', 'and', '&', 'of', 'for', 'in', 'at', 'on', 'a', 'an'}

def unicode_normalize(text):
    """Apply NFKC Unicode normalization."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKC', text)

def case_fold_and_whitespace(text):
    """Lowercase and normalize whitespaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_punctuation(text):
    """Remove standard punctuation to avoid matching issues."""
    # Replace common punctuation with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Re-condense spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_legal_suffixes(text):
    """Strip common legal suffixes from business names."""
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Re-condense spaces after removal
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def expand_business_abbreviations(text):
    """Expand common business abbreviations to their full forms."""
    for pattern, replacement in BUSINESS_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def apply_stemming(text):
    """Apply lightweight plural stemming for common business terms."""
    for pattern, replacement in STEMMING_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_numbers(text):
    """Numeric token normalization: strip leading zeros from numbers."""
    return re.sub(r'\b0+(\d+)\b', r'\1', text)

def remove_business_noise(text):
    """Business-specific noise removal: strip ids, secondary addresses."""
    # Strip (id: 12345) or id: 12345
    text = re.sub(r'\(?id:\s*\d+\)?', '', text, flags=re.IGNORECASE)
    # Strip secondary address unit designators
    text = re.sub(r'\b(?:po box|p\.o\. box|pmb|suite|ste|floor|fl|room|rm|apt)\s*#?\w+\b', '', text, flags=re.IGNORECASE)
    # Re-condense spaces after removal
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def expand_address_abbreviations(text):
    """Expand common address abbreviations to their full forms."""
    for pattern, replacement in ADDRESS_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_repeated_chars(text):
    """Collapse erroneously repeated characters (e.g., coooompany -> company)."""
    # Replaces 3 or more consecutive identical characters with just 1
    return re.sub(r'(.)\1{2,}', r'\1', text)

def remove_stopwords(text):
    """Remove common, low-value stopwords."""
    tokens = text.split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered_tokens)

def deduplicate_tokens(text):
    """Remove duplicate tokens while preserving order."""
    tokens = text.split()
    seen = set()
    dedup = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return " ".join(dedup)

def sort_tokens(text):
    """Sort tokens alphabetically for order-independent matching."""
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def extract_domain(url):
    """URL and domain normalization: Extracts core domain from URL."""
    if not isinstance(url, str):
         return ""
    url = url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return url

def advanced_clean_name(text, apply_token_sort=False):
    """Full pipeline for business names."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = unicode_normalize(text)
    text = case_fold_and_whitespace(text)
    text = remove_business_noise(text)
    text = remove_legal_suffixes(text)
    text = expand_business_abbreviations(text)
    text = apply_stemming(text)
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)
    text = normalize_numbers(text)
    text = remove_stopwords(text)
    text = deduplicate_tokens(text)
    if apply_token_sort:
        text = sort_tokens(text)
    return text

def advanced_clean_address(text, apply_token_sort=False):
    """Full pipeline for addresses."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = unicode_normalize(text)
    text = case_fold_and_whitespace(text)
    text = remove_business_noise(text)
    text = expand_address_abbreviations(text)
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)
    text = normalize_numbers(text)
    text = deduplicate_tokens(text)
    if apply_token_sort:
        text = sort_tokens(text)
    return text

def apply_advanced_preprocessing(df, apply_token_sort=False):
    """
    Applies the advanced text preprocessing pipeline to a dataframe
    containing 'business_name', 'business_address', and optionally 'country', 'url', 'phone'.
    Returns a pandas Series of the concatenated normalized text.
    """
    names = df['business_name'].apply(lambda x: advanced_clean_name(x, apply_token_sort))
    addresses = df['business_address'].apply(lambda x: advanced_clean_address(x, apply_token_sort))
    
    if "country" in df.columns:
         countries = df["country"].astype(object).fillna("").astype(str).str.lower().str.strip()
    else:
         countries = pd.Series([""] * len(df), index=df.index)

    combined = names + " | " + addresses + " | " + countries
    
    if "url" in df.columns:
         urls = df["url"].apply(extract_domain)
         combined = combined + " | " + urls
         
    if "phone" in df.columns:
         phones = df["phone"].astype(str).apply(lambda x: re.sub(r'\D', '', x) if pd.notna(x) else "")
         combined = combined + " | " + phones
         
    return combined
