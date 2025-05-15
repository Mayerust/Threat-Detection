import pandas as pd
import os
import sys

def load_dataset(path):
    if not os.path.exists(path):
        print(f"Error: File not found — {path}")
        return None
    return pd.read_csv(path)

def merge_datasets(train_path, malicious_path, output_path):
    print("Comment D1: Start")
    train_df = load_dataset(train_path)
    malicious_df = load_dataset(malicious_path)

    if train_df is None or malicious_df is None:
        print("Aborting due to missing dataset(s).")
        return

    # Ensure consistent column name
    if 'Type' in malicious_df.columns:
        malicious_df = malicious_df.rename(columns={'Type': 'attack_type'})

    print("Concatenating train and malicious datasets...")
    combined_df = pd.concat([train_df, malicious_df], ignore_index=True)

    print(f"Saving to {output_path}...") 
    
    #directory dk
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    combined_df.to_csv(output_path, index=False)
    print("Final merged dataset saved successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_datasets.py <processed_train.csv> <malicious_urls_processed.csv> <output_dir>")
        sys.exit(1)

    merge_datasets(sys.argv[1], sys.argv[2], sys.argv[3])
