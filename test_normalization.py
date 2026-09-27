import pandas as pd
from src.features.text_normalization import apply_advanced_preprocessing

if __name__ == "__main__":
    data = [
        # Original edge cases
        {"business_name": "Apple Inc.", "business_address": "1 Infinite Loop, Cupertino, CA", "country": "US", "url": "https://www.apple.com/mac", "phone": "(800) 555-1234"},
        {"business_name": "apple  corporation ", "business_address": "1 infinite loop cupertino ca", "country": "US", "url": "http://apple.com", "phone": "800-555-1234"},
        # New edge cases
        {"business_name": "The Bank of America & Co", "business_address": "123 Main St. Main St.", "country": "US", "url": None, "phone": None},
        {"business_name": "Bank America", "business_address": "123 main street", "country": "US", "url": None, "phone": None},
        {"business_name": "Miiiicrosofttt Corp", "business_address": "One Microsoft Way", "country": "US", "url": "www.microsoft.com", "phone": "123.456.7890"},
    ]
    df = pd.DataFrame(data)
    
    print("=== Original Data ===")
    print(df)
    
    print("\n=== Advanced Normalization (Default) ===")
    new_norm = apply_advanced_preprocessing(df, apply_token_sort=False)
    for t in new_norm:
        print(t)
        
    print("\n=== Advanced Normalization (With Token Sorting) ===")
    new_norm_sorted = apply_advanced_preprocessing(df, apply_token_sort=True)
    for t in new_norm_sorted:
        print(t)
