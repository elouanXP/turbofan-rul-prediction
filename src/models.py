import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


class Baseline:
    
    def fit(self, X, y):
        self.mean_ = y.mean()
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)


def plot_metrics_model(model, X_train, X_test, y_train, y_test, name="Model", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train  = mean_absolute_error(y_train, y_pred_train)
    r2_train   = r2_score(y_train, y_pred_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    r2_test   = r2_score(y_test, y_pred_test)

    print(f"{name} metrics")
    print("RMSE_train:", rmse_train, "RMSE_test:", rmse_test)
    print("MAE_train :", mae_train,  "MAE_test :", mae_test)
    print("R2_train  :", r2_train,   "R2_test  :", r2_test)

    ax.scatter(y_test, y_pred_test, alpha=0.3)
    ax.plot([0, y_test.max()], [0, y_test.max()], 'r--')
    ax.set_xlabel("True RUL")
    ax.set_ylabel(f"Predicted {name} RUL")
    ax.set_title(f"{name} Prediction")
    ax.grid()
    
def plot_unit_model(model, df, unit, features, scaler=None, name="Model", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()

    subset = df[df['unit_number'] == unit]

    X = subset[features]
    y = subset['RUL']

    if scaler is not None:
        X = scaler.transform(X)

    y_pred = model.predict(X)

    ax.plot(subset['time_cycles'], y, label='True RUL')
    ax.plot(subset['time_cycles'], y_pred, label='Pred RUL')
    ax.set_xlabel('cycle')
    ax.set_ylabel('RUL')
    ax.set_title(f'{name} Unit {unit}')
    ax.legend()
    ax.grid()
    
def get_feature_importance(model, X):
    importances = model.feature_importances_
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    
def plot_importance(model, X):
    feature_imp = get_feature_importance(model, X)
    feature_imp.plot(kind='barh')
    plt.grid()
    plt.show()
    plt.close()
    
def select_top_features(model, X, top_n=15):
    imp = get_feature_importance(model, X)
    cols = imp.index[:top_n]
    return cols

def evaluate_model(model, X_train, X_test, y_train, y_test, df, unit, features, scaler, name):
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    plot_metrics_model(model, X_train, X_test, y_train, y_test, name, axes[0])
    plot_unit_model(model, df, unit, features, scaler, name, axes[1])
    plt.tight_layout()
    plt.show()