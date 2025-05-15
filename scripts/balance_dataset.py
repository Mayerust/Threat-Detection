import pandas as pd
# import os
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer

# def balance_dataset(input_path, output_path):
#     # Load dataset
#     df = pd.read_csv(input_path, low_memory=False)
    
    

#     # Separate features and target variable
#     X = df.drop(columns=['attack_type'])
#     y = df['attack_type']
    
    

#     # Identify categorical columns
    
#     categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    

#     # Encode categorical columns
#     for col in categorical_cols:
#         le = LabelEncoder()
#         X[col] = le.fit_transform(X[col].astype(str))
        
        
        

#     # Handle missing values (NaNs)
#     imputer = SimpleImputer(strategy='mean')  # Replace NaN values with the mean
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    
    

#     # Apply SMOTE to balance the dataset
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)



#     # Combine resampled features and target variable
#     balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='attack_type')], axis=1)



#     # Save balanced dataset
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     balanced_df.to_csv(output_path, index=False)
#     print(f"Balanced dataset saved at: {output_path}")
    
    

# if __name__ == "__main__":
    
#     input_path = "/home/manas/threat_detection/data/processed/final_merged_dataset.csv"  
    
#     output_path = "/home/manas/threat_detection/data/processed/balanced_final_merged_dataset.csv"  
#     balance_dataset(input_path, output_path)

data = pd.read_csv("/home/manas/threat_detection/data/processed/balanced_final_merged_dataset.csv")
print(len(data.columns))
print(data["attack_type"].value_counts())
