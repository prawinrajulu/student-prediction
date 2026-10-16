"""
Validation utilities for the Student Performance Prediction System.
Provides input validation for individual student records and CSV datasets.
"""
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np


# Required columns for machine learning training and analysis
REQUIRED_TRAINING_COLUMNS = [
    "student_id",
    "attendance_percentage",
    "previous_semester_marks",
    "internal_assessment_marks",
    "assignment_marks",
    "study_hours",
    "final_exam_marks",
]

# Required input features for prediction
REQUIRED_PREDICTION_FEATURES = [
    "attendance_percentage",
    "previous_semester_marks",
    "internal_assessment_marks",
    "assignment_marks",
    "study_hours",
]

# Valid numerical constraints
VALIDATION_RANGES = {
    "attendance_percentage": (0.0, 100.0, "Attendance Percentage must be between 0% and 100%."),
    "previous_semester_marks": (0.0, 100.0, "Previous Semester Marks must be between 0 and 100."),
    "internal_assessment_marks": (0.0, 100.0, "Internal Assessment Marks must be between 0 and 100."),
    "assignment_marks": (0.0, 100.0, "Assignment Marks must be between 0 and 100."),
    "study_hours": (0.0, 24.0, "Daily Study Hours must be between 0 and 24 hours per day."),
    "final_exam_marks": (0.0, 100.0, "Final Examination Marks must be between 0 and 100."),
}


def validate_student_record(
    student_id: str,
    student_name: str,
    attendance: float,
    prev_marks: float,
    internal_marks: float,
    assignment_marks: float,
    study_hours: float,
    final_exam_marks: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates a student record before saving to data collection CSV.
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    # Student ID check
    cleaned_id = str(student_id).strip()
    if not cleaned_id:
        errors.append("Student ID is required and cannot be empty.")
    elif len(cleaned_id) > 30:
        errors.append("Student ID is too long (maximum 30 characters).")

    # Student Name check
    cleaned_name = str(student_name).strip()
    if not cleaned_name:
        errors.append("Student Name is required and cannot be empty.")
    elif len(cleaned_name) > 60:
        errors.append("Student Name is too long (maximum 60 characters).")

    # Range checks for numerical fields
    inputs = {
        "attendance_percentage": attendance,
        "previous_semester_marks": prev_marks,
        "internal_assessment_marks": internal_marks,
        "assignment_marks": assignment_marks,
        "study_hours": study_hours,
    }

    if final_exam_marks is not None:
        inputs["final_exam_marks"] = final_exam_marks

    for key, value in inputs.items():
        try:
            val_float = float(value)
            min_val, max_val, err_msg = VALIDATION_RANGES[key]
            if np.isnan(val_float) or val_float < min_val or val_float > max_val:
                errors.append(f"{err_msg} Received: {value}")
            elif key == "study_hours" and val_float > 16:
                # Warning rather than blocker, but check range
                pass
        except (ValueError, TypeError):
            errors.append(f"Invalid numerical format for {key.replace('_', ' ').title()}: '{value}'")

    return (len(errors) == 0, errors)


def validate_prediction_inputs(
    attendance: float,
    prev_marks: float,
    internal_marks: float,
    assignment_marks: float,
    study_hours: float,
) -> Tuple[bool, List[str]]:
    """
    Validates feature inputs for real-time model prediction.
    """
    errors: List[str] = []
    inputs = {
        "attendance_percentage": attendance,
        "previous_semester_marks": prev_marks,
        "internal_assessment_marks": internal_marks,
        "assignment_marks": assignment_marks,
        "study_hours": study_hours,
    }

    for key, val in inputs.items():
        try:
            val_float = float(val)
            min_val, max_val, err_msg = VALIDATION_RANGES[key]
            if np.isnan(val_float) or val_float < min_val or val_float > max_val:
                errors.append(f"{err_msg} (Given: {val})")
        except (ValueError, TypeError):
            errors.append(f"'{key.replace('_', ' ').title()}' must be a valid number.")

    return (len(errors) == 0, errors)


def validate_csv_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validates an uploaded or historical CSV dataset for model training.
    Returns:
        (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("The uploaded dataset is empty.")
        return False, errors, warnings

    # Normalize column names (lowercase, stripped)
    normalized_cols = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    df_renamed = df.rename(columns=normalized_cols)

    # Check required columns
    missing_cols = [c for c in REQUIRED_TRAINING_COLUMNS if c not in df_renamed.columns]
    if missing_cols:
        errors.append(f"Missing required columns in dataset: {', '.join(missing_cols)}")
        return False, errors, warnings

    # Check minimum rows
    if len(df_renamed) < 15:
        errors.append(
            f"Dataset has only {len(df_renamed)} records. At least 15 records are required for a meaningful train-test split."
        )
    elif len(df_renamed) < 50:
        warnings.append(
            f"Dataset has only {len(df_renamed)} records. For reliable machine learning evaluation, 100+ records are recommended."
        )

    # Numerical validation on features and target
    for col in REQUIRED_PREDICTION_FEATURES + ["final_exam_marks"]:
        non_null = df_renamed[col].dropna()
        if len(non_null) == 0:
            errors.append(f"Column '{col}' has no valid numerical values.")
            continue

        try:
            series = pd.to_numeric(non_null, errors="coerce")
            nan_count = series.isna().sum()
            if nan_count > 0:
                warnings.append(f"Column '{col}' contains {nan_count} non-numerical entries that will be handled.")

            valid_vals = series.dropna()
            min_val, max_val, msg = VALIDATION_RANGES[col]
            out_of_bounds = ((valid_vals < min_val) | (valid_vals > max_val)).sum()
            if out_of_bounds > 0:
                warnings.append(f"Column '{col}' contains {out_of_bounds} values outside the typical range [{min_val}, {max_val}].")
        except Exception as e:
            errors.append(f"Error parsing column '{col}': {str(e)}")

    # Check target variance
    target_series = pd.to_numeric(df_renamed["final_exam_marks"], errors="coerce").dropna()
    if len(target_series) > 1 and target_series.std() < 1e-4:
        errors.append("Target column 'final_exam_marks' has constant values (zero variance). Regression cannot be trained.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings
