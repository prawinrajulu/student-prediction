"""
Module 4: Machine Learning Model Training.
Trains, evaluates, and compares:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Calculates MAE, RMSE, and R2 score using identical train-test splits.
Persists the best-performing model pipeline and training metadata using Joblib.
"""
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Backward compatibility for scikit-learn < 1.4 where root_mean_squared_error was not available
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return float(mean_squared_error(y_true, y_pred) ** 0.5)

from modules.preprocessing import preprocess_data, FEATURE_COLUMNS, TARGET_COLUMN

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "best_student_model.joblib")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.joblib")


def train_and_compare_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Trains Linear Regression, Decision Tree, and Random Forest models on identical data splits.
    Evaluates each model using MAE, RMSE, and R2.
    Selects the best model based on minimum Mean Absolute Error (MAE).
    """
    errors: List[str] = []

    # Preprocess dataset
    prep_result, prep_errors = preprocess_data(
        df=df,
        test_size=test_size,
        random_state=random_state,
        apply_scaling=False,  # Scaler will be integrated inside Pipeline for Linear Regression
    )

    if prep_errors or prep_result is None:
        return None, prep_errors

    X_train = prep_result.X_train
    X_test = prep_result.X_test
    y_train = prep_result.y_train
    y_test = prep_result.y_test

    # Define candidate pipelines
    # Linear Regression benefits from standard scaling
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    # Tree models are invariant to monotonic feature scaling, but keeping same interface
    dt_pipeline = Pipeline([
        ("model", DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=random_state,
        )),
    ])

    rf_pipeline = Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
        )),
    ])

    candidates = {
        "Linear Regression": lr_pipeline,
        "Decision Tree Regressor": dt_pipeline,
        "Random Forest Regressor": rf_pipeline,
    }

    results = {}
    predictions = {}

    for name, pipeline in candidates.items():
        # Fit on training set only
        pipeline.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipeline.predict(X_test)
        # Examination marks are strictly bounded [0, 100]
        y_pred_clamped = np.clip(y_pred, 0.0, 100.0)
        predictions[name] = y_pred_clamped

        # Calculate metrics
        mae = float(mean_absolute_error(y_test, y_pred_clamped))
        rmse = float(root_mean_squared_error(y_test, y_pred_clamped))
        r2 = float(r2_score(y_test, y_pred_clamped))

        results[name] = {
            "pipeline": pipeline,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "predictions": y_pred_clamped,
        }

    # Summary table
    metrics_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE (Lower is Better)": [results[k]["mae"] for k in results],
        "RMSE (Lower is Better)": [results[k]["rmse"] for k in results],
        "R² Score (Higher is Better)": [results[k]["r2"] for k in results],
    }).sort_values(by="MAE (Lower is Better)").reset_index(drop=True)

    # Determine best model based on lowest MAE (robust to outliers and intuitive for grade marks)
    best_model_name = metrics_df.iloc[0]["Model"]
    best_pipeline = results[best_model_name]["pipeline"]
    best_metrics = {
        "mae": results[best_model_name]["mae"],
        "rmse": results[best_model_name]["rmse"],
        "r2": results[best_model_name]["r2"],
    }

    # Save best model to disk
    save_trained_model(
        pipeline=best_pipeline,
        model_name=best_model_name,
        metrics=best_metrics,
        features=FEATURE_COLUMNS,
        imputer=prep_result.imputer,
        dataset_records=len(df),
    )

    output = {
        "metrics_df": metrics_df,
        "results": results,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "best_pipeline": best_pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "imputer": prep_result.imputer,
        "preprocessing_summary": prep_result.summary,
        "warnings": prep_result.warnings,
    }

    return output, []


def save_trained_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    features: List[str],
    imputer: Any,
    dataset_records: int,
) -> None:
    """Persists model pipeline and metadata safely with Joblib."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(pipeline, MODEL_FILE)

    metadata = {
        "model_name": model_name,
        "metrics": metrics,
        "feature_names": features,
        "imputer": imputer,
        "dataset_records": dataset_records,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
    }
    joblib.dump(metadata, METADATA_FILE)


def load_saved_model() -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    """
    Loads active model pipeline and associated metadata from disk.
    Performs runtime verification to ensure the loaded model is functional
    and compatible with the current Python environment.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    try:
        pipeline = joblib.load(MODEL_FILE)
        metadata = joblib.load(METADATA_FILE)

        # Integrity verification
        if not hasattr(pipeline, "predict") or not isinstance(metadata, dict):
            return None, None

        # Sanity check dry-run with expected feature schema to catch version discrepancies
        dummy_input = pd.DataFrame([{
            "attendance_percentage": 75.0,
            "previous_semester_marks": 70.0,
            "internal_assessment_marks": 72.0,
            "assignment_marks": 80.0,
            "study_hours": 3.5,
        }])
        _ = pipeline.predict(dummy_input)

        return pipeline, metadata
    except Exception:
        # If unpickling or execution fails due to version incompatibility, signal failure cleanly
        return None, None


def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Generates a comparison bar chart of Mean Absolute Error (MAE) across all 3 models.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    models = metrics_df["Model"].tolist()
    maes = metrics_df["MAE (Lower is Better)"].tolist()

    # Highlight best (lowest MAE) with vibrant emerald green, others in soft slate blue
    min_mae = min(maes)
    colors = ["#2ca02c" if m == min_mae else "#4c78a8" for m in maes]

    bars = ax.bar(models, maes, color=colors, width=0.5, edgecolor="black", linewidth=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"MAE: {height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylim(0, max(maes) * 1.35)
    ax.set_title("Model Comparison: Mean Absolute Error (Lower = Better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Machine Learning Algorithm")
    ax.set_ylabel("Mean Absolute Error (Exam Marks)")
    fig.tight_layout()
    return fig
