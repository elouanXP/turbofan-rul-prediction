# Smart Predictive Maintenance — NASA CMAPSS Turbofan Engine

[![Python](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)

Predictive maintenance system for turbofan engines using the **NASA CMAPSS dataset**.  
The model predicts the **Remaining Useful Life (RUL)** of an engine — i.e. how many operational cycles remain before failure — enabling condition-based maintenance decisions.

---

## Results

| Model | RMSE (test) | MAE (test) | R² (test) |
|-------|-------------|------------|-----------|
| Baseline | 39.92 | 35.26 | -0.000 |
| Linear Regression | 19.00 | 15.39 | 0.773 |
| Random Forest | 15.90 | 10.92 | 0.841 |
| RF Tuned | 15.51 | 10.71 | 0.849 |
| XGBoost | 16.98 | 11.48 | 0.819 |
| XGBoost Tuned | 15.53 | 11.18 | 0.849 |

**NASA Benchmark (test_FD001.txt — full train)** : RMSE = 17.34 · MAE = 12.10 · R² = 0.806

> Evaluated on the official NASA test set using the full training data — consistent with published results for classical ML approaches on FD001.

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
│   └── 04_business_conclusion.ipynb    # NASA scoring function, operational risk, physical interpretation
│
├── outputs/
│   ├── models/               # Saved pipelines (.joblib)
│   └── plots/                # All generated figures (.png)
│
├── src/
│   ├── config.py             # Paths and hyperparameters
│   ├── preprocessing.py      # RUL computation, feature engineering, filtering
│   └── model.py              # Training utilities, evaluation, MLflow experiment tracking
│
├── requirements.txt
└── README.md
```

---

## Dataset

**CMAPSS** (Commercial Modular Aero-Propulsion System Simulation) — NASA Ames Research Center.  
Simulates the degradation of a **turbofan engine high-pressure compressor** from healthy state to failure.

- **FD001** : 100 training engines · 100 test engines · single operating condition · single fault mode
- Each engine starts with unknown initial wear and develops a fault that grows until system failure
- 21 sensor measurements per cycle (temperature, pressure, speed, fuel flow...)
- **Objective** : predict the number of remaining operational cycles at the last observed measurement

[Download the dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## Methodology

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Column naming from CMAPSS documentation
- Sensor visualization per engine
- Lifetime distribution analysis
- Variance and correlation analysis

### 2. Feature Engineering (`02_feature_engineering.ipynb`)
- **RUL computation** with clipping at 120 cycles (standard in CMAPSS literature)
- **Low-variance feature removal** — sensors with near-zero variance carry no information
- **Temporal features** per engine : rolling mean (window=5), rolling std, first difference
- **Correlation filtering** — features with |corr(RUL)| ≤ 0.1 removed

### 3. Modelling (`03_model.ipynb`)
- **Unit-based train/test split** — no engine appears in both train and test (prevents data leakage)
- **GroupKFold cross-validation** — folds built by engine unit to prevent leakage during tuning
- **sklearn Pipeline** — scaler + model in a single object, guaranteeing consistent preprocessing at inference
- **MLflow tracking** — all runs logged with parameters, metrics and artifacts
- **Feature selection** by model importance before hyperparameter tuning (top 15 features)
- **Final evaluation** on official NASA test set after retraining on full training data

### 4. Business Analysis (`04_business_conclusion.ipynb`)
- **NASA scoring function** — asymmetric penalty: late predictions (missed failures) penalized more than early predictions (unnecessary maintenance)
- **Operational risk classification** — engines categorized as safe / early warning / danger
- **Physical sensor interpretation** — top features mapped to actual turbofan degradation mechanisms
- **Limitations and next steps** — honest assessment of model boundaries

---

## Quickstart

### Prerequisites
```bash
git clone https://github.com/elouanXP/smart_predictive_maintenance
cd smart_predictive_maintenance
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data
Download the CMAPSS dataset and place `CMAPSSdata.zip` in `data/raw/`.

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

## Key Design Decisions

**Why split by engine unit, not randomly?**  
A random row-level split would place cycles from the same engine in both train and test — the model would partially memorize each engine's behavior rather than generalize. Splitting by unit ensures the test set contains engines never seen during training.

**Why clip RUL at 120 cycles?**  
At high RUL values (engine is healthy), the exact remaining life is operationally irrelevant — no maintenance decision depends on whether an engine has 200 or 280 cycles remaining. Clipping reduces noise in the target and focuses the model on the degradation phase that matters for scheduling.

**Why use GroupKFold for cross-validation?**  
Standard KFold would randomly assign cycles to folds, again creating leakage between engines. GroupKFold ensures each engine appears in exactly one fold as validation data.

---

## Author

**elouanXP** · [GitHub](https://github.com/elouanXP/smart_predictive_maintenance)

---

*Dataset: NASA CMAPSS — Saxena, A., Goebel, K., Simon, D., & Ecker, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. PHM Conference.*