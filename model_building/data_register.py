from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# *** UPDATE THIS TO YOUR HF USERNAME ***
repo_id = "sandy1916/tourism-package-data"
repo_type = "dataset"

# Initialise API client using the HF_TOKEN secret
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the dataset repo exists; create if absent
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, exist_ok=True)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the data folder to the HF dataset hub
api.upload_folder(
    folder_path="Advanced-Machine-Learning-and-MLOps/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Data successfully registered at {repo_id}")
