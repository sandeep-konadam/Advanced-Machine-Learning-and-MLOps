import os
import mlflow
import joblib
from datasets import load_dataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi, create_repo

# *** UPDATE THIS TO YOUR HF USERNAME ***
dataset_repo = "your-username/tourism-package-data"
model_repo = "your-username/tourism-prediction-model"

train_data = load_dataset(dataset_repo + "-train", token=os.getenv("HF_TOKEN"))['train'].to_pandas()
test_data = load_dataset(dataset_repo + "-test", token=os.getenv("HF_TOKEN"))['train'].to_pandas()

X_train, y_train = train_data.drop('ProdTaken', axis=1), train_data['ProdTaken']
X_test, y_test = test_data.drop('ProdTaken', axis=1), test_data['ProdTaken']

mlflow.set_experiment("Tourism_Package_Prediction")
best_acc, best_model = 0, None

for lr in [0.01, 0.1]:
    for depth in [3, 5]:
        with mlflow.start_run():
            model = XGBClassifier(learning_rate=lr, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)
            
            if acc > best_acc:
                best_acc, best_model = acc, model

# Register Model in Hugging Face
model_filename = "xgboost_tourism_model.joblib"
joblib.dump(best_model, model_filename)
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=model_repo, repo_type="model")
except:
    create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=model_repo,
    repo_type="model"
)
print("Model trained, tracked, and pushed successfully.")
