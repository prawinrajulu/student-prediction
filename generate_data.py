"""
Script to generate synthetic academic dataset for student performance prediction.
Uses reproducible random seed and plausible academic distributions.
"""
import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    student_ids = [f"STU{1001 + i}" for i in range(num_samples)]

    # Generate realistic base distributions
    # Attendance: beta distribution skewed towards 75-90%
    attendance = np.clip(np.random.beta(a=5, b=2, size=num_samples) * 55 + 45, 45.0, 99.5)

    # Previous semester marks: normal distribution around 68, std 14
    prev_marks = np.clip(np.random.normal(loc=68, scale=13, size=num_samples), 35.0, 98.0)

    # Internal assessment marks: correlated moderately with previous semester marks
    internal_noise = np.random.normal(loc=0, scale=8, size=num_samples)
    internal_marks = np.clip(0.65 * prev_marks + 0.35 * (attendance * 0.7 + 15) + internal_noise, 32.0, 99.0)

    # Assignment marks: generally slightly higher, with some variance
    assignment_noise = np.random.normal(loc=2, scale=7, size=num_samples)
    assignment_marks = np.clip(0.4 * prev_marks + 0.3 * attendance + 20 + assignment_noise, 40.0, 100.0)

    # Study hours: lognormal or gamma distribution, average around 3.5 - 4.5 hours
    study_hours = np.clip(np.random.gamma(shape=3.5, scale=1.1, size=num_samples), 0.5, 9.5)

    # Final examination marks:
    # Influenced by internals (weight 0.32), previous semester (0.26), attendance (0.14),
    # assignment marks (0.12), study hours (1.2 per hr), plus realistic non-linear noise
    noise = np.random.normal(loc=0, scale=4.8, size=num_samples)

    final_exam_marks = (
        0.32 * internal_marks +
        0.26 * prev_marks +
        0.14 * attendance +
        0.12 * assignment_marks +
        1.25 * study_hours +
        noise - 3.5
    )

    # Realistic occasional anomalies (e.g., student studied hard but had bad exam day or vice versa)
    anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    final_exam_marks[anomaly_indices] += np.random.choice([-12, -9, 8, 11], size=len(anomaly_indices))

    final_exam_marks = np.clip(final_exam_marks, 20.0, 99.0)

    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance_percentage": np.round(attendance, 1),
        "previous_semester_marks": np.round(prev_marks, 1),
        "internal_assessment_marks": np.round(internal_marks, 1),
        "assignment_marks": np.round(assignment_marks, 1),
        "study_hours": np.round(study_hours, 1),
        "final_exam_marks": np.round(final_exam_marks, 1),
    })

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = generate_synthetic_data(num_samples=250, seed=42)
    output_path = os.path.join("data", "student_performance.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records in {output_path}")

    # Generate sample initial student_data.csv with 5 sample records for immediate demo
    initial_students = pd.DataFrame([
        {
            "student_id": "STU2001",
            "student_name": "Aarav Sharma",
            "attendance_percentage": 88.5,
            "previous_semester_marks": 78.0,
            "internal_assessment_marks": 82.5,
            "assignment_marks": 85.0,
            "study_hours": 4.5,
            "submitted_at": "2026-10-15 10:30:00"
        },
        {
            "student_id": "STU2002",
            "student_name": "Priya Patel",
            "attendance_percentage": 92.0,
            "previous_semester_marks": 86.5,
            "internal_assessment_marks": 90.0,
            "assignment_marks": 94.0,
            "study_hours": 5.5,
            "submitted_at": "2026-10-15 11:15:00"
        },
        {
            "student_id": "STU2003",
            "student_name": "Rohan Verma",
            "attendance_percentage": 64.0,
            "previous_semester_marks": 52.0,
            "internal_assessment_marks": 48.0,
            "assignment_marks": 55.0,
            "study_hours": 2.0,
            "submitted_at": "2026-10-15 12:00:00"
        },
        {
            "student_id": "STU2004",
            "student_name": "Ananya Iyer",
            "attendance_percentage": 76.0,
            "previous_semester_marks": 71.0,
            "internal_assessment_marks": 68.5,
            "assignment_marks": 74.0,
            "study_hours": 3.5,
            "submitted_at": "2026-10-15 14:20:00"
        },
        {
            "student_id": "STU2005",
            "student_name": "Vikram Singh",
            "attendance_percentage": 52.0,
            "previous_semester_marks": 44.0,
            "internal_assessment_marks": 40.0,
            "assignment_marks": 48.0,
            "study_hours": 1.5,
            "submitted_at": "2026-10-15 15:05:00"
        }
    ])
    student_data_path = os.path.join("data", "student_data.csv")
    initial_students.to_csv(student_data_path, index=False)
    print(f"Initialized {len(initial_students)} records in {student_data_path}")

    # Create .gitkeep in models
    gitkeep_path = os.path.join("models", ".gitkeep")
    with open(gitkeep_path, "w") as f:
        f.write("")
    print("Created models/.gitkeep")
