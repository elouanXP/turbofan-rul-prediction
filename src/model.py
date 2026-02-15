import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import mlflow
import mlflow.sklearn
import pandas as pd
from src.config import (
    RANDOM_STATE, 
    TEST_SIZE,
    ROOT
)


def split_by_unit(
    df: pd.DataFrame,
    unit_column:str = 'unit_number',
    test_size:float = TEST_SIZE,
    random_state:int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split a dataset into train and test sets based on unique unit identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing all samples.
    unit_column : str, default='unit_number'
        Name of the column identifying the unit/group.
    test_size : float, default=TEST_SIZE
        Proportion of units assigned to the test set
    random_state : int, default=RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    train_units : list
        List of unit identifiers used in training.
    test_units : list
        List of unit identifiers used in testing.
    train_df : pd.DataFrame
        Training subset.
    test_df : pd.DataFrame
        Testing subset.
    """
    df = df.copy()
    units = df[unit_column].unique()

    train_units, test_units = train_test_split(
        units,
        test_size=test_size,
        random_state=random_state
    )

    train_df = df[df[unit_column].isin(train_units)]
    test_df = df[df[unit_column].isin(test_units)]

    return train_df, test_df, train_units, test_units


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = None,
    drop_columns: list[str] = None,
    scaler = None
) -> Tuple[
    pd.DataFrame, 
    pd.DataFrame, 
    pd.DataFrame, 
    pd.DataFrame,
    pd.Series, 
    pd.Series,
    StandardScaler
]:
    """
    Prepare feature matrices and target vectors for training and testing.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Testing dataset.
    target_column : str, default='RUL'
        Name of the target variable.
    drop_columns : list of str, default=['unit_number','time_cycles','RUL']
        Additional columns to drop from features.
    scaler, default=StandardScaler

    Returns
    -------
    X_train : pd.DataFrame
        Training feature matrix.
    X_train_scaled : pd.DataFrame
        Training scaled feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    X_test_scaled : pd.DataFrame
        Testing scaled feature matrix.
    y_train : pd.Series
        Training target vector.
    y_test : pd.Series
        Testing target vector.
    scaler
    """
    target_column = target_column or 'RUL'
    drop_columns = drop_columns or ['unit_number','time_cycles','RUL']
    scaler = scaler or StandardScaler()
    
    X_train = train_df.drop(columns=drop_columns).copy()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = train_df[target_column].copy()

    X_test = test_df.drop(columns=drop_columns).copy()
    X_test_scaled  = scaler.transform(X_test)
    y_test = test_df[target_column].copy()

    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, scaler


class Baseline(BaseEstimator, RegressorMixin):
    """
    Reference model predicting the mean of the training target.
    """
    def fit(self, X, y):
        self.mean_ = y.mean()
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)


def plot_metrics_model(
    model, 
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    name="Model", 
    ax=None
) -> dict:
    
    """
    Compute and display regression metrics  for train and test sets.

    Metrics computed:
    - RMSE
    - MAE
    - R²

    Plot predicted vs true values for the test set.

    Parameters
    ----------
    model :
        Trained regression model.
    X_train, X_test : array
        Training and testing feature matrices.
    y_train, y_test : array
        True target values.
    name : str
        Model name.
    ax : matplotlib axis, optional
        Axis for plotting.

    Returns
    -------
    dict
        Dictionary containing RMSE, MAE and R² for train and test sets.
        Keys: rmse_train, rmse_test, mae_train, mae_test, r2_train, r2_test.
    """
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
    
    metrics = {
    'rmse_train': rmse_train, 'rmse_test': rmse_test,
    'mae_train': mae_train,   'mae_test': mae_test,
    'r2_train': r2_train,     'r2_test': r2_test
    }
    return metrics

    
def plot_unit_model(
    model, 
    df, 
    unit, 
    features, 
    scaler=None, 
    name="Model", 
    ax=None
):
    
    """
    Plot true vs predicted RUL over time for a specific unit.

    Parameters
    ----------
    model :
        Trained regression model.
    df : pd.DataFrame
        Full dataset including unit_number, time_cycles and RUL.
    unit : int
        Unit identifier to visualize.
    features : list
        Feature column names used by the model.
    scaler : fitted scaler, optional
        Scaler used for feature normalization.
    name : str
        Model name.
    ax : matplotlib axis, optional
        Axis for plotting.
    """   
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
    """
    Compute and sort feature importances from tree-based models.

    Parameters
    ----------
    model :
        Model with attribute `feature_importances_`.
    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    pd.Series
        Sorted feature importance values.
    """
    importances = model.feature_importances_
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    
def plot_importance(model, X, save_path=None):
    """
    Plot feature importances for tree-based models.

    Parameters
    ----------
    model :
        Model with attribute `feature_importances_`.
    X : pd.DataFrame
        Feature dataframe.
    save_path : str
        Path to save model plots
    """
    feature_imp = get_feature_importance(model, X)
    feature_imp.plot(kind='barh')
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    

def select_top_features(model, X, top_n=15):
    """
    Select the top N most important features based on model importance.

    Parameters
    ----------
    model :
        Model with attribute `feature_importances_`.
    X : pd.DataFrame
        Feature dataframe.
    top_n : int
        Number of top features to keep.

    Returns
    -------
    list
        List of selected feature names.
    """
    imp = get_feature_importance(model, X)
    cols = imp.index[:top_n]
    return cols


def evaluate_model(
    model, 
    X_train,
    X_test, 
    y_train, 
    y_test, 
    df, 
    unit, 
    features,
    name,
    scaler=None, 
    save_path=None
) -> dict:
    
    """
    Evaluate a regression model and visualize its performance.

    This function:
    - Computes standard regression metrics
    - Plots predicted vs true values
    - Visualizes RUL prediction over time for a specific unit

    Parameters
    ----------
    model :
        Trained regression model.
    X_train, X_test : array
        Feature matrices.
    y_train, y_test : array
        Target vectors.
    df : pd.DataFrame
        Full dataset for unit-level visualization.
    unit : int
        Unit identifier to visualize.
    features : list
        Feature column names used by the model.
    scaler : default = None
        Optional scaler used for normalization.
    name : str
        Model name.
    save_path : str
        Path to save model plots
    Returns
    -------
    dict
        Dictionary containing RMSE, MAE and R² for train and test sets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    metrics=plot_metrics_model(model, X_train, X_test, y_train, y_test, name, axes[0])
    plot_unit_model(model, df, unit, features, scaler, name, axes[1])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    return metrics


def run_experiment(
    experiment_name: str,
    run_name: str,
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df: pd.DataFrame,
    unit: int,
    features: list,
    save_path: None
) -> dict:
    """
    Train a pipeline, log parameters and metrics to MLflow,
    and return evaluation metrics.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment (groups multiple runs).
    run_name : str
        Name of the individual run.
    pipeline : sklearn.pipeline.Pipeline
        Pipeline containing preprocessing steps and model.
    X_train : pd.DataFrame
        Training features (unscaled)
    X_test : pd.DataFrame
        Testing features (unscaled)
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Testing target values.
    df : pd.DataFrame
        Full dataset used for visualization purposes.
    unit : int
        Unit identifier to visualize.
    features : list of str
        Feature columns used by the model.
    save_path : str
        Path to save model plots.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics (RMSE, MAE, R² for train and test sets).
    """

    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            model=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            df=df,
            unit=unit,
            features=features,
            name=run_name,
            save_path=save_path
        )

        mlflow.log_params(pipeline.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name=run_name, registered_model_name=run_name)

    return metrics