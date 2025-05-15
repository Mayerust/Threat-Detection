# AI-Driven Threat Detection System

## Overview

This repository implements a comprehensive pipeline for detecting threats—both phishing URLs and network-based attacks—using classic datasets (NSL-KDD) and real-world malicious URL collections. The workflow covers data ingestion, preprocessing, merging, balancing, and automated model evaluation via multiple classifiers. The model is trained with the final absolutely balanced dataset containing 500,000 datapoint (250,000 benign and 250,000 attack).

## Prerequisites

* **Python**: 3.8+
* **Libraries**:

  * pandas (`pip install pandas`)
  * NumPy (`pip install numpy`)
  * scikit-learn (`pip install scikit-learn`)
  * imbalanced-learn (`pip install imbalanced-learn`)
  * joblib (for model persistence, `pip install joblib`)

## Repository Structure

```
├── data/
│   ├── raw/
│   │   ├── malicious_urls.csv               # Original URL threat dataset
│   │   ├── KDDTrain+.txt                     # NSL-KDD training data (no headers)
│   │   ├── KDDTest+.txt                      # NSL-KDD test set
│   │   ├── KDDTest-21.txt                    # NSL-KDD alternative test set
│   │   └── (optional `.arff` files)
│   ├── processed/
│   │   ├── malicious_urls_processed.csv      # Labeled URL dataset (attack_type column)
│   │   ├── processed_train.csv               # NSL-KDD training with headers and normalization
│   ├── merged/
│   │   ├── final_merged_dataset.csv          # Concatenation of processed NSL-KDD and URL data
│   │   └── balanced_final_merged_dataset.csv # SMOTE- or manual-balanced dataset
│   
│
├── scripts/
│   ├── preprocess_malicious_urls.py          # Clean & label URL dataset
│   ├── preprocess_nsl_kdd.py                 # Assign headers, map labels, normalize NSL-KDD
│   ├── merge_datasets.py                     # Merge processed CSVs into one
│   ├── balance_dataset.py                    # Inspect/balance dataset (SMOTE code commented)
│   └── automl.py                             # Subsample, preprocess, & evaluate multiple 
│
├── README.md
└── LICENSE
```
## Running the complete project

 * If you want to run the complete project, make sure to extract all '.gz' and '.zip' files, containing datasets (>25MB) in '.csv' format.


## Data Pipeline

1. **Raw Data** (`data/raw/`)

   * `malicious_urls.csv`: contains URL samples labeled via a numeric column (e.g., `Label` or `Type`).
   * `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTest-21.txt`: standard NSL-KDD files without headers.

2. **Processing** (`scripts/preprocess_*.py`)

   * **URLs**: identify the correct label column, map `1 → malicious`, `0 → benign`, drop raw label, save `malicious_urls_processed.csv`.
   * **NSL-KDD**: assign human-readable column names, map `'normal.' → normal`, others → `attack`, drop unused features, apply min–max scaling, output `processed_train.csv`.

3. **Merging** (`scripts/merge_datasets.py`)

   * Rename URL dataset column `Type` → `attack_type` for uniformity.
   * Concatenate NSL-KDD and URL DataFrames.
   * Output `final_merged_dataset.csv`.

4. **Balancing** (`scripts/balance_dataset.py`)

   * (Commented) demonstrates how to use SMOTE for oversampling.
   * Current script loads `balanced_final_merged_dataset.csv` and prints feature counts and class distribution.

5. **AutoML Evaluation** (`scripts/automl.py`)

   * Subsample equal numbers of `benign` and `attack` from the balanced dataset.
   * Drop constant or high-cardinality columns (e.g., URLs).
   * Compute correlations and display top features.
   * Train and cross-validate a suite of models: Logistic Regression, SVM, Nearest Centroid, Perceptron, KNN, Ridge, MLP, etc.
   * Print per-model CV accuracy, validation accuracy, and runtime.

## Usage

1. **Install dependencies**:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn joblib
   ```

2. **Run preprocessing**:

   ```bash
   python scripts/preprocess_malicious_urls.py
   python scripts/preprocess_nsl_kdd.py
   ```

3. **Merge datasets**:

   ```bash
   python scripts/merge_datasets.py data/processed/processed_train.csv data/processed/malicious_urls_processed.csv data/merged/final_merged_dataset.csv
   ```

4. **Balance dataset (optional)**:

   * Uncomment and configure SMOTE section in `scripts/balance_dataset.py`, then run:

   ```bash
   python scripts/balance_dataset.py
   ```

5. **Run AutoML evaluation**:

   ```bash
   python scripts/automl.py
   ```

## Directory Configuration

* Modify paths at the top of each script to match your file system if needed.
* Ensure `data/raw/`, `data/processed/`, and `data/merged/` directories exist.

## Results & Artifacts

* **Processed CSVs**: `data/processed/*.csv`
* **Merged Datasets**: `data/merged/*.csv`
* **Balanced Dataset**: `data/merged/balanced_final_merged_dataset.csv`
* **Model Outputs**: can be saved under `outputs/` (not currently enabled but recommended).

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "Add XYZ"`)
4. Push to the branch (`git push origin feature/XYZ`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
