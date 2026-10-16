"""
Module 1: Student Data Collection.
Manages student record storage, validation, retrieval, searching, CSV export, and deletion.
"""
import os
from datetime import datetime
from typing import Tuple, List, Optional
import pandas as pd
from utils.validation import validate_student_record

DATA_FILE = os.path.join("data", "student_data.csv")

COLUMNS = [
    "student_id",
    "student_name",
    "attendance_percentage",
    "previous_semester_marks",
    "internal_assessment_marks",
    "assignment_marks",
    "study_hours",
    "submitted_at",
]


def ensure_data_file() -> None:
    """Ensures that the student data file and parent directory exist."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(DATA_FILE, index=False)


def load_student_records() -> pd.DataFrame:
    """Loads all collected student records from CSV."""
    ensure_data_file()
    try:
        df = pd.read_csv(DATA_FILE)
        # Ensure correct string types
        if not df.empty and "student_id" in df.columns:
            df["student_id"] = df["student_id"].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def add_student_record(
    student_id: str,
    student_name: str,
    attendance: float,
    prev_marks: float,
    internal_marks: float,
    assignment_marks: float,
    study_hours: float,
) -> Tuple[bool, str]:
    """
    Validates and appends a student record to student_data.csv.
    """
    is_valid, errors = validate_student_record(
        student_id=student_id,
        student_name=student_name,
        attendance=attendance,
        prev_marks=prev_marks,
        internal_marks=internal_marks,
        assignment_marks=assignment_marks,
        study_hours=study_hours,
    )

    if not is_valid:
        return False, " | ".join(errors)

    ensure_data_file()
    df = load_student_records()

    # Check for duplicate student_id
    if not df.empty and "student_id" in df.columns:
        if str(student_id).strip().upper() in df["student_id"].str.upper().values:
            return False, f"A student with ID '{student_id.strip()}' already exists. Use a unique Student ID."

    new_record = {
        "student_id": str(student_id).strip().upper(),
        "student_name": str(student_name).strip(),
        "attendance_percentage": round(float(attendance), 2),
        "previous_semester_marks": round(float(prev_marks), 2),
        "internal_assessment_marks": round(float(internal_marks), 2),
        "assignment_marks": round(float(assignment_marks), 2),
        "study_hours": round(float(study_hours), 2),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    new_df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    new_df.to_csv(DATA_FILE, index=False)
    return True, f"Student record for '{student_name.strip()}' ({student_id.strip().upper()}) saved successfully!"


def delete_student_record(student_id: str) -> Tuple[bool, str]:
    """
    Deletes a student record by student ID.
    """
    ensure_data_file()
    df = load_student_records()

    if df.empty:
        return False, "No student records available to delete."

    target_id = str(student_id).strip().upper()
    initial_len = len(df)
    filtered_df = df[df["student_id"].str.upper() != target_id]

    if len(filtered_df) == initial_len:
        return False, f"Student ID '{student_id}' not found."

    filtered_df.to_csv(DATA_FILE, index=False)
    return True, f"Record for Student ID '{target_id}' deleted successfully."


def search_student_records(query: str) -> pd.DataFrame:
    """
    Searches student records by student_id or student_name (case-insensitive).
    """
    df = load_student_records()
    if df.empty or not query.strip():
        return df

    q = query.strip().lower()
    matches = df[
        df["student_id"].astype(str).str.lower().str.contains(q)
        | df["student_name"].astype(str).str.lower().str.contains(q)
    ]
    return matches
