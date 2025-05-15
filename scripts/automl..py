import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer

# Import classifiers:
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def load_subsample(path, sample_size=5000):
    print("Debug C: Starting AutoML")
    df = pd.read_csv(path, low_memory=False)
    
    
    if 'attack_type' not in df.columns:
        if 'label' in df.columns:
            df.rename(columns={'label': 'attack_type'}, inplace=True)
            print("Renamed column 'label' to 'attack_type'.")
        else:
            raise KeyError("Expected column 'attack_type' (or alternative 'label') was not found. Available columns: " + str(df.columns.tolist()))
        
        
        
        
        
    
   
    groups = []
    for group_name, group_df in df.groupby('attack_type'):
        n = min(len(group_df), sample_size // 2)
        groups.append(group_df.sample(n=n, random_state=42))
    df = pd.concat(groups).reset_index(drop=True)
    
    print("Debug C: Preprocessing data...")
    
   
    df['target'] = df['attack_type'].map({'benign': 0, 'attack': 1})
    
    
    
    
    #  duplicates.
    dupes = df.duplicated().sum()
    print(f"\n Duplicate rows in data: {dupes}")
    
    #  constant columns.
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    df.drop(columns=const_cols, inplace=True)
    print(f" Constant columns ({len(const_cols)}): {const_cols}")
    
    
    #  high-cardinality string columns.
    drop_cols = ['url']  
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    print(f"Dropped columns: {drop_cols}")
    
    
    
    
    
    
        
    
    # Correlation-based feature selection.
    df_corr = df.drop(columns=['attack_type'])  
    top_corr = df_corr.corr()['target'].drop('target').abs().sort_values(ascending=False)
    print("\n Top 10 correlated features:")
    print(top_corr.head(10))
    
    
    
    
    # Class distribution.
    print("\n Class distribution in subsample:")
    print(df['attack_type'].value_counts(normalize=True).rename("proportion"))
    
    return df

def preprocess(df):
    X = df.drop(columns=['attack_type', 'target'])
    y = df['target']
    
    
    
    
    # Impute missing values.
    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)
    
    
    
    
    return X, y
def evaluate_models(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    models = {
        "Logistic": LogisticRegression(max_iter=500, n_jobs=-1, solver='lbfgs'),
        "SGD": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "NearestCentroid": NearestCentroid(),
        "SVC": SVC(kernel='rbf', C=1.0, random_state=42),
        "NuSVC": NuSVC(nu=0.5, kernel='rbf', random_state=42),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Ridge": RidgeClassifier(),
        "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    }
    
    print("\nModel             | CV Acc  | Val Acc | Time(s)")
    for name, model in models.items():
        start = time.time()
        cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        elapsed = round(time.time() - start, 2)
        print(f"{name:<16}| {cv_score:.4f}  | {val_score:.4f}  | {elapsed}")

def main():
    df = load_subsample("../data/processed/balanced_final_merged_dataset.csv")
    X, y = preprocess(df)
    evaluate_models(X, y)

if __name__ == "__main__":
    main()