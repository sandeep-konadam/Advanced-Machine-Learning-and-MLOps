import os
import mlflow
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# *** UPDATE THIS TO YOUR HF USERNAME ***
dataset_repo = "sandy1916/tourism-package-data"
model_repo   = "sandy1916/tourism-prediction-model"

# Load prepared train/test splits from HF Hub
train_data = load_dataset(dataset_repo + "-train", token=os.getenv("HF_TOKEN"))["train"].to_pandas()
test_data  = load_dataset(dataset_repo + "-test",  token=os.getenv("HF_TOKEN"))["train"].to_pandas()

X_train, y_train = train_data.drop("ProdTaken", axis=1), train_data["ProdTaken"]
X_test,  y_test  = test_data.drop("ProdTaken",  axis=1), test_data["ProdTaken"]
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Define candidate models with hyperparameter grids
candidates = [
    {
        "name": "DecisionTree",
        "model_class": DecisionTreeClassifier,
        "params_grid": [
            {"max_depth": 4, "min_samples_split": 5},
            {"max_depth": 6, "min_samples_split": 10},
        ],
    },
    {
        "name": "RandomForest",
        "model_class": RandomForestClassifier,
        "params_grid": [
            {"n_estimators": 100, "max_depth": 5},
            {"n_estimators": 200, "max_depth": 7},
        ],
    },
    {
        "name": "GradientBoosting",
        "model_class": GradientBoostingClassifier,
        "params_grid": [
            {"learning_rate": 0.05, "n_estimators": 100, "max_depth": 3},
            {"learning_rate": 0.10, "n_estimators": 150, "max_depth": 4},
        ],
    },
    {
        "name": "XGBoost",
        "model_class": XGBClassifier,
        "params_grid": [
            {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 100},
            {"learning_rate": 0.10, "max_depth": 5, "n_estimators": 150},
        ],
    },
]

# Run experiments and track with MLflow
mlflow.set_experiment("Tourism_Package_Prediction")
best_acc, best_model, best_model_name = 0, None, ""

for candidate in candidates:
    for params in candidate["params_grid"]:
        with mlflow.start_run(run_name=f"{candidate['name']}_{params}"):
            # Build and train model
            model = candidate["model_class"](random_state=42, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Evaluate
            acc     = accuracy_score(y_test, preds)
            f1      = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, preds)

            # Log to MLflow
            mlflow.log_param("model", candidate["name"])
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            print(f"{candidate['name']} | params={params} | acc={acc:.4f} | f1={f1:.4f}")

            # Track best model
            if acc > best_acc:
                best_acc, best_model = acc, model
                best_model_name = candidate["name"]

print(f"\nBest model: {best_model_name} with accuracy={best_acc:.4f}")

# Serialise best model
model_filename = "tourism_best_model.joblib"
joblib.dump(best_model, model_filename)

# Register best model on HF Model Hub
api = HfApi(token=os.getenv("HF_TOKEN"))
try:
    api.repo_info(repo_id=model_repo, repo_type="model")
    print(f"Model repo '{model_repo}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=model_repo, repo_type="model", private=False, exist_ok=True)
    print(f"Model repo '{model_repo}' created.")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=model_repo,
    repo_type="model",
)
print(f"Best model ({best_model_name}) registered at {model_repo}")
