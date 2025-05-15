import pandas as pd
import os

def process_malicious_urls(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: The file at {input_path} does not exist.")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    label_col = None
    for col in df.columns:
        if col.lower() in ['type', 'label', 'class', 'target']:
            label_col = col
            break

    if not label_col:
        print("Error: No recognizable label column found (like 'Type', 'Label', etc.)")
        return

    df['attack_type'] = df[label_col].apply(lambda x: 'malicious' if x == 1 else 'benign')
    df = df.drop(columns=[label_col])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")

if __name__ == "__main__":
    input_path = "/home/manas/threat_detection/data/raw/malicious_urls.csv"
    output_path = "/home/manas/threat_detection/data/processed/malicious_urls_processed.csv"
    process_malicious_urls(input_path, output_path)