"""
Module 2: Data Preprocessing.
Loads, inspects, cleans, imputes, and splits academic performance datasets.
Follows best ML practices: avoids data leakage by fitting transformers strictly on training data.
"""
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "attendance_percentage",
    "previous_semester_marks",
    "internal_assessment_marks",
    "assignment_marks",
    "study_hours",
]

TARGET_COLUMN = "final_exam_marks"


class PreprocessingResult:
    """Container for preprocessed data and preprocessing summary."""
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        cleaned_df: pd.DataFrame,
        summary: Dict[str, Any],
        imputer: SimpleImputer,
        scaler: Optional[StandardScaler] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cleaned_df = cleaned_df
        self.summary = summary
        self.imputer = imputer
        self.scaler = scaler
        self.warnings = warnings or []


def load_dataset(file_path_or_buffer) -> pd.DataFrame:
    """Loads CSV dataset and standardizes column headers."""
    df = pd.read_csv(file_path_or_buffer)
    # Standardize column headers: trim, lowercase, underscores
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates diagnostic statistics on missing values, data types, and duplicates."""
    total_records = len(df)
    total_columns = len(df.columns)
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    duplicate_count = int(df.duplicated().sum())

    return {
        "total_records": total_records,
        "total_columns": total_columns,
        "missing_counts": missing_counts,
        "total_missing_values": total_missing,
        "duplicate_records": duplicate_count,
        "columns": list(df.columns),
    }


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_scaling: bool = False,
) -> Tuple[Optional[PreprocessingResult], List[str]]:
    """
    Executes full data cleaning and train-test splitting pipeline.
    Prevents data leakage by computing imputation statistics strictly on X_train.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("Provided dataframe is empty.")
        return None, errors

    # Check for feature and target presence
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {', '.join(missing_features)}")

    if TARGET_COLUMN not in df.columns:
        errors.append(f"Missing required target column: '{TARGET_COLUMN}'")

    if errors:
        return None, errors

    total_loaded = len(df)
    initial_missing = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().to_dict()
    total_initial_missing = sum(initial_missing.values())

    # Step 1: Remove duplicates
    cleaned_df = df.copy()
    duplicates_removed = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # Step 2: Validate target variable - drop rows with missing target
    cleaned_df[TARGET_COLUMN] = pd.to_numeric(cleaned_df[TARGET_COLUMN], errors="coerce")
    target_missing = cleaned_df[TARGET_COLUMN].isnull().sum()
    if target_missing > 0:
        cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        warnings.append(f"Dropped {target_missing} rows where target '{TARGET_COLUMN}' was missing or non-numeric.")

    # Clamp target to realistic bounds [0, 100]
    cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].clip(0.0, 100.0)

    # Check sample size
    records_available = len(cleaned_df)
    if records_available < 15:
        errors.append(
            f"Only {records_available} records remain after cleaning. Minimum 15 records are required for training and evaluation."
        )
        return None, errors
    elif records_available < 50:
        warnings.append(
            f"Dataset has {records_available} records. Evaluation metrics may have higher variance on smaller datasets."
        )

    # Step 3: Separate features and target
    X = cleaned_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = cleaned_df[TARGET_COLUMN].copy()

    # Step 4: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 5: Impute missing numerical feature values using Median fit only on X_train (No Data Leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )

    # Optional scaling
    scaler = None
    if apply_scaling:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imputed),
            columns=FEATURE_COLUMNS,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imputed),
            columns=FEATURE_COLUMNS,
            index=X_test.index,
        )
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = X_train_imputed
        X_test_final = X_test_imputed

    # Summary dictionary
    summary = {
        "total_records_loaded": total_loaded,
        "missing_values_detected": total_initial_missing,
        "missing_per_column": initial_missing,
        "duplicates_removed": duplicates_removed,
        "records_available_after_cleaning": records_available,
        "selected_features": FEATURE_COLUMNS,
        "target_variable": TARGET_COLUMN,
        "train_set_size": len(X_train),
        "test_set_size": len(X_test),
        "test_size_ratio": test_size,
        "imputation_strategy": "Median Imputation (fit on train split only)",
        "scaling_applied": apply_scaling,
    }

    result = PreprocessingResult(
        X_train=X_train_final,
        X_test=X_test_final,
        y_train=y_train,
        y_test=y_test,
        cleaned_df=cleaned_df,
        summary=summary,
        imputer=imputer,
        scaler=scaler,
        warnings=warnings,
    )

    return result, errors
