import pandas as pd
import numpy as np

"""
This code implements an undersampling method with noise-based data augmentation loosely mased off of SMOTE concepts to ensure that fraud cases make up exactly 1% of total transactions.
The dataset that was chosen did not reflect the real world, as both fraud and non-fraud cases were balanced 50-50.
To counteract this, undersample.py undersamples the non-fraud transactions, then in a SMOTE-like manner (synthetic augmentation), adds a small amount of noise to randomly selected
fraud cases, then creates a dataset where fraud cases make up exactly 1% of the data
"""

data = pd.read_csv("dataset/creditcard_2023.csv")

# Separates fraud and non-fraud cases
fraud_cases = data[data["Class"] == 1]
non_fraud_cases = data[data["Class"] == 0]

# Calculates the total number of transactions needed to make fraud exactly 1% of the dataset
total_transactions = int(len(non_fraud_cases) / 0.99) 
n_new_frauds = total_transactions - len(non_fraud_cases) - len(fraud_cases)

if n_new_frauds > 0:
    
    # Randomly selects existing fraud samples to create synthetic ones
    synthetic_fraud_samples = fraud_cases.sample(n=n_new_frauds, replace=True, random_state=42).copy()

    feature_columns = synthetic_fraud_samples.columns.difference(["Class"])  # Get feature column names
    noise = np.random.normal(0, 0.01, synthetic_fraud_samples[feature_columns].shape)  # Small noise for variation
    synthetic_fraud_samples[feature_columns] += noise  # Add noise only to feature columns (Not Class column)

    # Ensures all generated fraud cases are labeled as 1
    synthetic_fraud_samples["Class"] = 1

    # Merge original fraud cases, synthetic frauds, and non-fraud cases
    balanced_data = pd.concat([non_fraud_cases, fraud_cases, synthetic_fraud_samples])
else:
    # If no new fraud cases are needed, downsample fraud cases to exactly 1% of the total dataset
    target_fraud_count = int(len(non_fraud_cases) * 0.01)
    fraud_cases = fraud_cases.sample(n=target_fraud_count, random_state=42)
    balanced_data = pd.concat([non_fraud_cases, fraud_cases])

# Shuffle dataset
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Save new dataset
balanced_data.to_csv("dataset/creditcard_realistic.csv", index=False)

print(f"New dataset created with fraud cases making up exactly 1% of total transactions.")