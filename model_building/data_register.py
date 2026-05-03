import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

# *** UPDATE THIS TO YOUR HF USERNAME ***
repo_id = "your-username/tourism-package-data"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if space exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except:
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    print(f"Created space '{repo_id}'.")

# Load local data and push to hub
df = pd.read_csv('Advanced-Machine-Learning-and-MLOps/data/tourism.csv')
hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub(repo_id, token=os.getenv("HF_TOKEN"))

print(f"Data successfully registered at {repo_id}")
