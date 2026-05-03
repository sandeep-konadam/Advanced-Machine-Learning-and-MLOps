from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
# *** UPDATE THIS TO YOUR HF USERNAME AND TARGET SPACE NAME ***
SPACE_REPO = "your-username/Tourism-Prediction-Space" 

api.upload_folder(
    folder_path="Advanced-Machine-Learning-and-MLOps/deployment",     
    repo_id=SPACE_REPO,          
    repo_type="space",                      
    path_in_repo="",                          
)
print("Deployment folder pushed to Hugging Face Space.")
