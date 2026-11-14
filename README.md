# ChurnScope — Customer Churn Prediction Dashboard

A complete **Streamlit dashboard** for an educational customer churn prediction model. It is built on the same synthetic dataset and logistic regression pipeline as the original `src/main.py` project.

> **Demo only:** The data is synthetic, and metrics are not evidence of real-world predictive performance. Do not use for customer decisions.

## Features
- Interactive customer input form (monthly charges, tenure, support tickets, contract type)
- Estimated churn probability and a 50% risk-label display threshold
- Accuracy, precision, recall, F1, ROC-AUC on a held-out test set
- Confusion matrix and ROC curve
- Dataset explorer, contract churn chart, charges histogram, CSV download
- Cached data/model for fast navigation

## Run on Windows / VS Code

1. Extract this ZIP and open **the folder containing `app.py`** in VS Code.
2. Open **Terminal → New Terminal**.
3. Check Python: `py --version`
4. Install dependencies: `py -m pip install -r requirements.txt`
5. Start the web app: `py -m streamlit run app.py`
6. Visit **http://localhost:8501** in Chrome or Edge. Use **Ctrl+C** in the terminal to stop.

If `8501` is occupied, Streamlit will offer another local port. If Windows cannot find `py`, use `python` instead.

## Run original command-line project

`py src/main.py` — regenerates `data/customers_demo.csv`, `results/metrics.json`, and `screenshots/churn_evaluation.png`.

## Files

- `app.py`: Streamlit web dashboard
- `src/main.py`: original reproducible model script and demo-data generator
- `data/customers_demo.csv`: synthetic sample dataset
- `screenshots/churn_evaluation.png`: chart from original script
- `screenshots/dashboard_preview.png`: illustrative dashboard preview from synthetic data
- `results/metrics.json`: saved metrics from original script
- `requirements.txt`: Python dependencies

## Model & limitations

Logistic regression with standardized numeric features and one-hot encoded contract type. Stratified 75/25 train/test split, `random_state=42`. Probability is **not calibrated or validated** for real customers. No cross-validation, temporal validation, production monitoring, fairness assessment, or external validation. Synthetic features and outcomes are simplified. Dashboard predictions retrain from the included dataset at startup; do not interpret the demo risk label as a business recommendation.

## GitHub

Upload all project files except your virtual environment, caches, and secrets. Screenshot images are generated from synthetic demo data and must be labeled accordingly.
