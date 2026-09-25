import json
from pathlib import Path
import re

def is_non_ascii(text):
    return not text.isascii()

def has_missing_address(text):
    # Format is "name | address | country"
    parts = text.split(" | ")
    if len(parts) >= 2:
        return parts[1].strip() == ""
    return False

def count_duplicates(strings):
    # Check if there are identical strings in the list
    seen = set()
    dupes = 0
    for s in strings:
        # Strip ID if it exists? Wait, the strings are just "name | address | country"
        # Since we use those strings, they might be identical.
        if s in seen:
            dupes += 1
        seen.add(s)
    return dupes > 0

def analyze():
    base_dir = Path(__file__).resolve().parents[2]
    shared_errors_file = base_dir / "reports" / "shared_errors.json"
    
    with open(shared_errors_file, "r", encoding="utf-8") as f:
        errors = json.load(f)
        
    print(f"Total Shared Errors: {len(errors)}")
    
    # Categories
    non_ascii_count = 0
    missing_address_count = 0
    duplicate_targets = 0
    cross_language_suspected = 0
    
    for err in errors:
        query = err['query']
        true_matches = err['true_matches']
        
        # Check non-ascii in query or any true match
        has_na = is_non_ascii(query) or any(is_non_ascii(tm) for tm in true_matches)
        if has_na:
            non_ascii_count += 1
            # Simple heuristic for cross-language: if non-ascii chars are outside typical latin-1 extended
            # Let's just assume non-ascii indicates cross-language for Indian businesses
            if 'india' in query.lower():
                cross_language_suspected += 1
                
        # Check missing address
        has_ma = has_missing_address(query) or any(has_missing_address(tm) for tm in true_matches)
        if has_ma:
            missing_address_count += 1
            
        # Check ID recycling (duplicate text representations in true matches)
        # Actually in error compare, we output the raw text (which doesn't have IDs).
        if count_duplicates(true_matches):
            duplicate_targets += 1
            
    print(f"\n--- Trap Analysis on {len(errors)} Failed Examples ---")
    print(f"Trap 1 & 5 (Non-ASCII / Cross-Language): {non_ascii_count} (In India: {cross_language_suspected})")
    print(f"Trap 2 (ID Recycling / Exact Duplicates in Ground Truth): {duplicate_targets}")
    print(f"Trap 3 (Missing Addresses in Query or Ground Truth): {missing_address_count}")
    print(f"Trap 4 (France): 0 (Test set only)")
    
    # Calculate percentage explained
    # Let's count how many errors fall into AT LEAST one of these traps
    explained = 0
    for err in errors:
        q = err['query']
        tm = err['true_matches']
        
        na = is_non_ascii(q) or any(is_non_ascii(t) for t in tm)
        ma = has_missing_address(q) or any(has_missing_address(t) for t in tm)
        dup = count_duplicates(tm)
        
        if na or ma or dup:
            explained += 1
            
    print(f"\nTotal Errors Explained by Traps: {explained} / {len(errors)} ({explained/len(errors)*100:.1f}%)")
    print(f"Unexplained Errors: {len(errors) - explained}")

if __name__ == "__main__":
    analyze()
