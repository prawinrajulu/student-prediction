"""
Module 5: Student Performance Prediction.
Applies trained Scikit-learn pipelines to predict expected final exam marks.
Computes configurable letter grades, academic performance categories,
factor analysis, and automated recommendations.
"""
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from modules.model_training import load_saved_model, train_and_compare_models
from modules.recommendations import generate_academic_recommendations
from utils.validation import validate_prediction_inputs

DEFAULT_GRADE_THRESHOLDS = [
    ("A", 90.0, 100.0, "Outstanding / Excellent Mastery"),
    ("B", 80.0, 89.99, "Very Good / High Competence"),
    ("C", 70.0, 79.99, "Good / Above Average"),
    ("D", 60.0, 69.99, "Satisfactory / Average"),
    ("E", 50.0, 59.99, "Pass / Minimum Acceptable"),
    ("F", 0.0, 49.99, "Fail / Inadequate"),
]

DEFAULT_CATEGORY_THRESHOLDS = {
    "At Risk": (0.0, 49.99, "#d9534f", "Student is in danger of failing; proactive academic intervention required."),
    "Average": (50.0, 74.99, "#f0ad4e", "Student is performing satisfactorily but has clear room for score elevation."),
    "Good": (75.0, 100.0, "#5cb85c", "Student demonstrates strong grasp and steady academic standing."),
}


def calculate_grade(marks: float, custom_thresholds: Optional[List[Tuple]] = None) -> Tuple[str, str]:
    """Calculates letter grade and description based on marks out of 100."""
    thresholds = custom_thresholds or DEFAULT_GRADE_THRESHOLDS
    clamped = max(0.0, min(100.0, float(marks)))

    for grade, low, high, desc in thresholds:
        if low <= clamped <= high:
            return grade, desc
    return ("F", "Fail / Inadequate") if clamped < 50.0 else ("A", "Outstanding")


def calculate_category(marks: float, custom_categories: Optional[Dict] = None) -> Tuple[str, str, str]:
    """
    Calculates category: At Risk, Average, or Good.
    Returns: (category_name, hex_color, description)
    """
    categories = custom_categories or DEFAULT_CATEGORY_THRESHOLDS
    clamped = max(0.0, min(100.0, float(marks)))

    if clamped < 50.0:
        return "At Risk", categories["At Risk"][2], categories["At Risk"][3]
    elif clamped < 75.0:
        return "Average", categories["Average"][2], categories["Average"][3]
    else:
        return "Good", categories["Good"][2], categories["Good"][3]


def predict_student_performance(
    attendance: float,
    prev_marks: float,
    internal_marks: float,
    assignment_marks: float,
    study_hours: float,
    custom_grade_thresholds: Optional[List[Tuple]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Validates input features, loads saved model (or trains baseline if none exists),
    predicts final exam marks, computes grade, category, explanation, and recommendations.
    """
    # Step 1: Input Validation
    is_valid, errors = validate_prediction_inputs(
        attendance=attendance,
        prev_marks=prev_marks,
        internal_marks=internal_marks,
        assignment_marks=assignment_marks,
        study_hours=study_hours,
    )
    if not is_valid:
        return None, errors

    # Step 2: Load Model Pipeline
    pipeline, metadata = load_saved_model()

    if pipeline is None:
        # Fallback: check if student_performance.csv exists to auto-train initial baseline model
        csv_path = "data/student_performance.csv"
        try:
            if pd.io.common.file_exists(csv_path):
                df_hist = pd.read_csv(csv_path)
                train_out, train_err = train_and_compare_models(df_hist)
                if train_out:
                    pipeline, metadata = load_saved_model()
        except Exception:
            pass

    if pipeline is None:
        errors.append("No trained model found. Please train a machine learning model on the 'Model Training' page first.")
        return None, errors

    # Step 3: Format features into DataFrame matching trained columns
    input_data = pd.DataFrame([{
        "attendance_percentage": float(attendance),
        "previous_semester_marks": float(prev_marks),
        "internal_assessment_marks": float(internal_marks),
        "assignment_marks": float(assignment_marks),
        "study_hours": float(study_hours),
    }])

    # Step 4: Run prediction
    try:
        raw_prediction = float(pipeline.predict(input_data)[0])
    except Exception as e:
        errors.append(f"Model prediction failed: {str(e)}")
        return None, errors

    # Strictly clamp marks to valid exam range [0.0, 100.0]
    predicted_marks = round(float(np.clip(raw_prediction, 0.0, 100.0)), 2)

    # Step 5: Grade and Performance Category
    grade, grade_desc = calculate_grade(predicted_marks, custom_grade_thresholds)
    category, cat_color, cat_desc = calculate_category(predicted_marks)

    # Step 6: Factor Explanation
    # Analyze which factors are strong or weak relative to cohort benchmarks
    factors = []
    if internal_marks >= 75.0:
        factors.append(f"Strong internal assessment score ({internal_marks:.1f}%) contributes positively.")
    elif internal_marks < 50.0:
        factors.append(f"Low internal assessment score ({internal_marks:.1f}%) pulls the estimated mark down.")

    if attendance >= 80.0:
        factors.append(f"Reliable attendance ({attendance:.1f}%) reinforces consistent academic engagement.")
    elif attendance < 75.0:
        factors.append(f"Sub-optimal attendance ({attendance:.1f}%) limits expected examination readiness.")

    if study_hours >= 4.0:
        factors.append(f"Solid daily study routine ({study_hours:.1f} hrs/day) aids retention.")
    elif study_hours < 2.0:
        factors.append(f"Limited study commitment ({study_hours:.1f} hrs/day) indicates need for structured study.")

    if not factors:
        factors.append("Overall metrics align closely with median academic benchmarks.")

    explanation_text = " ".join(factors)

    # Step 7: Rule-Based Recommendations
    recommendations = generate_academic_recommendations(
        attendance=attendance,
        prev_marks=prev_marks,
        internal_marks=internal_marks,
        assignment_marks=assignment_marks,
        study_hours=study_hours,
        predicted_marks=predicted_marks,
    )

    result = {
        "predicted_marks": predicted_marks,
        "grade": grade,
        "grade_description": grade_desc,
        "category": category,
        "category_color": cat_color,
        "category_description": cat_desc,
        "model_name": metadata.get("model_name", "Best Trained Model") if metadata else "Selected ML Model",
        "model_metrics": metadata.get("metrics", {}) if metadata else {},
        "trained_at": metadata.get("trained_at", "N/A") if metadata else "N/A",
        "explanation": explanation_text,
        "recommendations": recommendations,
        "inputs": {
            "attendance": attendance,
            "prev_marks": prev_marks,
            "internal_marks": internal_marks,
            "assignment_marks": assignment_marks,
            "study_hours": study_hours,
        },
    }

    return result, []
