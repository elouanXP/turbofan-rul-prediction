# Turbofan RUL Prediction -- NASA CMAPSS

[![Python](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.9.0-blue.svg)](https://mlflow.org/)

End-to-end predictive maintenance system for turbofan engines using the NASA CMAPSS dataset. The model predicts the **Remaining Useful Life (RUL)** of a turbofan engine high-pressure compressor (number of operational cycles remaining before failure) in order to improve maintenance decisions.

---

## Results

| Model | RMSE (test) | MAE (test) | R² (test) |
|-------|-------------|------------|-----------|
| Baseline | 39.92 | 35.26 | 0.00 |
| Linear Regression | 19.00 | 15.39 | 0.77 |
| Random Forest | 15.90 | 10.92 | 0.84 |
| **RF Tuned** | **15.51** | **10.71** | **0.85** |
| XGBoost | 16.98 | 11.48 | 0.82 |
| XGBoost Tuned | 15.53 | 11.18 | 0.85 |

NASA Benchmark (Random Forest Tuned - full train) = RMSE = 17.34 · MAE = 12.10 · R² = 0.806 · NASA Score = 907.2

![plot](outputs/plots/04_operational_risk_FD001.png)

---

## Project Structure

```
smart_predictive_maintenance/
│
├── data/
│   ├── raw/                  # Original NASA CMAPSS files
│   └── processed/            # Cleaned and feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA, sensor visualization, variance analysis
│   ├── 02_feature_engineering.ipynb    # RUL computation, feature creation, correlation filtering
│   ├── 03_model.ipynb                  # Model training, tuning, comparison, NASA benchmark
│   └── 04_business_conclusion.ipynb    # NASA scoring function, operational risk
│
├── outputs/
│   ├── models/               # Saved pipelines (.joblib)
│   └── plots/                # All generated figures (.png)
│
├── src/
│   ├── config.py             # Paths and variables
│   ├── preprocessing.py      # RUL computation, feature engineering, filtering
│   └── model.py              # Training utilities, evaluation, MLflow experiment tracking
│
├── requirements.txt
└── README.md
```
---

## Quickstart

### Prerequisites
```bash
git clone https://github.com/elouanXP/turbofan-rul-prediction
cd turbofan-rul-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data
[Download the CMAPSS dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place `CMAPSSdata.zip` in `data/raw/`.


### Run notebooks
Execute notebooks in order from the `notebooks/` directory:
```
01_data_exploration.ipynb
02_feature_engineering.ipynb
03_model.ipynb
04_business_conclusion.ipynb
```

### MLflow UI
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` to explore all tracked experiments and compare runs.

---

## Technical Stack

| Category | Tools |
|----------|-------|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Visualization | matplotlib, seaborn |
| Model persistence | joblib |

---

## Author

[elouanXP](https://github.com/elouanXP/)