from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# *** UPDATE THIS TO YOUR HF USERNAME AND SPACE NAME ***
SPACE_REPO = "sandy1916/Tourism-Prediction-Space"

# Push all deployment files to the Hugging Face Space
api.upload_folder(
    folder_path="deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files pushed to Hugging Face Space: {SPACE_REPO}")
