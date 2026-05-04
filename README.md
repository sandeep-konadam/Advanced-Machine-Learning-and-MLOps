# Advanced Machine Learning and MLOps
### Tourism Package Prediction — End-to-End MLOps Pipeline

---

## Business Context

"Visit with Us" is a leading travel company aiming to improve customer targeting for its newly introduced **Wellness Tourism Package**. The manual approach to identifying potential buyers is inconsistent and time-consuming. This project implements a fully automated MLOps pipeline that predicts whether a customer will purchase the package — enabling data-driven, scalable marketing decisions.

---

## Project Structure

```
Advanced-Machine-Learning-and-MLOps/
├── data/                          # Raw dataset (tourism.csv)
├── model_building/
│   ├── data_register.py           # Registers raw data on HF Dataset Hub
│   ├── prep.py                    # Data cleaning, encoding, train/test split
│   └── train.py                   # Model training, MLflow tracking, best model registration
├── deployment/
│   ├── app.py                     # Streamlit prediction app
│   ├── Dockerfile                 # Container configuration for HF Space
│   └── requirements.txt           # App dependencies
├── hosting/
│   └── hosting.py                 # Pushes deployment files to HF Space
├── requirements.txt               # Pipeline dependencies for GitHub Actions
└── .github/
    └── workflows/
        └── pipeline.yml           # CI/CD pipeline definition
```

---

## MLOps Pipeline

The pipeline is fully automated via **GitHub Actions** and triggers on every push to `main`.

```
Push to main
     ↓
register-dataset   →   Uploads tourism.csv to Hugging Face Dataset Hub
     ↓
data-prep          →   Cleans, encodes, splits data — pushes train/test splits to HF
     ↓
model-training     →   Trains 4 models, tracks with MLflow, registers best model on HF
     ↓
deploy-hosting     →   Pushes Streamlit app + Dockerfile to Hugging Face Space
```

---

## Models Trained

Four algorithms are evaluated with hyperparameter tuning. All runs are tracked using **MLflow**. The best model by accuracy is automatically selected and registered.

| Algorithm | Parameters Tuned |
|---|---|
| Decision Tree | max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth |
| Gradient Boosting | learning_rate, n_estimators, max_depth |
| XGBoost | learning_rate, max_depth, n_estimators |

---

## Hugging Face Assets

| Asset | Type | URL |
|---|---|---|
| Raw Dataset | Dataset | `sandy1916/tourism-package-data` |
| Train Split | Dataset | `sandy1916/tourism-package-data-train` |
| Test Split | Dataset | `sandy1916/tourism-package-data-test` |
| Best Model | Model | `sandy1916/tourism-prediction-model` |
| Streamlit App | Space | [sandy1916/Tourism-Prediction-Space](https://huggingface.co/spaces/sandy1916/Tourism-Prediction-Space) |

---

## Setup Instructions

### 1. Prerequisites
- GitHub account with this repository forked/cloned
- Hugging Face account with a **Write** access token
- Hugging Face Space created: `your-username/Tourism-Prediction-Space` (Docker + Streamlit, Public)

### 2. Add GitHub Secret
Go to **Settings → Secrets and variables → Actions → New repository secret**
- Name: `HF_TOKEN`
- Value: your Hugging Face Write token

### 3. Run the Notebook
Open `setup_final.ipynb` in **Google Colab**, upload `tourism.csv` to the `data/` folder, and run all cells top to bottom. The final cells push everything to GitHub and trigger the pipeline automatically.

### 4. Verify
Go to the **Actions** tab in this repository to monitor all four pipeline jobs running in sequence.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model Training | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Data & Model Registry | Hugging Face Hub |
| App Framework | Streamlit |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Notebook Environment | Google Colab |

---

## Output

- **GitHub Repository**: https://github.com/sandeep-konadam/Advanced-Machine-Learning-and-MLOps
- **Streamlit App**: https://huggingface.co/spaces/sandy1916/Tourism-Prediction-Space
