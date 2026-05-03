import os
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# *** UPDATE THIS TO YOUR HF USERNAME ***
repo_id = "your-username/tourism-package-data"

# Load dataset
dataset = load_dataset(repo_id, token=os.getenv("HF_TOKEN"))
df = dataset['train'].to_pandas()

# Clean Data
columns_to_drop = ['Unnamed: 0', 'CustomerID']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

df_encoded = pd.get_dummies(df_clean, drop_first=True)

# Split 
X = df_encoded.drop('ProdTaken', axis=1)
y = df_encoded['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Upload splits back to Hub
Dataset.from_pandas(pd.concat([X_train, y_train], axis=1)).push_to_hub(repo_id + "-train", token=os.getenv("HF_TOKEN"))
Dataset.from_pandas(pd.concat([X_test, y_test], axis=1)).push_to_hub(repo_id + "-test", token=os.getenv("HF_TOKEN"))
print("Data Preparation and Splitting complete.")
