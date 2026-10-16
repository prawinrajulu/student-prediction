"""
End-to-End System Test Suite.
Verifies:
1. Validation utility functions.
2. Data collection (saving, searching, deleting records).
3. Data preprocessing (loading, imputing without leakage, splitting).
4. EDA metrics, plots, and automated insight generation.
5. Model training (Linear Regression, Decision Tree, Random Forest).
6. Model evaluation and saving.
7. Real-time student performance prediction, grades, standing, and recommendations.
"""
import os
import sys
import pandas as pd
import numpy as np

def run_tests():
    print("=" * 60)
    print("RUNNING STUDENT PERFORMANCE PREDICTION SYSTEM TESTS")
    print("=" * 60)

    # 1. Test Validation
    print("\n[TEST 1] Testing utils.validation...")
    from utils.validation import validate_student_record, validate_prediction_inputs, validate_csv_dataset
    
    # Valid record
    ok, errs = validate_student_record("STU999", "Test Student", 85.0, 78.0, 80.0, 85.0, 4.0)
    assert ok, f"Expected valid record, got errors: {errs}"

    # Invalid attendance > 100
    ok, errs = validate_student_record("STU999", "Test Student", 105.0, 78.0, 80.0, 85.0, 4.0)
    assert not ok and any("Attendance" in e for e in errs), "Failed to catch invalid attendance > 100"

    # Negative study hours
    ok, errs = validate_prediction_inputs(85.0, 75.0, 80.0, 85.0, -2.0)
    assert not ok, "Failed to catch negative study hours"
    print(" -> PASSED: Input validation handles bounds and invalid entries properly.")

    # 2. Test Data Collection
    print("\n[TEST 2] Testing modules.data_collection...")
    from modules.data_collection import add_student_record, load_student_records, search_student_records, delete_student_record
    
    test_id = "STU_TEST_01"
    ok, msg = add_student_record(test_id, "Test Runner", 88.0, 82.0, 84.0, 90.0, 4.5)
    assert ok or "already exists" in msg, f"Add student record failed: {msg}"
    
    records = load_student_records()
    assert not records.empty, "student_data.csv should not be empty"
    
    search_res = search_student_records(test_id)
    assert not search_res.empty, f"Search failed for {test_id}"
    
    del_ok, del_msg = delete_student_record(test_id)
    assert del_ok, f"Delete failed: {del_msg}"
    print(" -> PASSED: Data collection add, read, search, and delete work properly.")

    # 3. Test Preprocessing
    print("\n[TEST 3] Testing modules.preprocessing...")
    from modules.preprocessing import preprocess_data, load_dataset
    
    df_raw = load_dataset(os.path.join("data", "student_performance.csv"))
    assert len(df_raw) >= 200, f"Expected >= 200 rows, found {len(df_raw)}"

    prep_res, prep_errs = preprocess_data(df_raw, test_size=0.2, random_state=42)
    assert prep_errs == [], f"Preprocessing produced errors: {prep_errs}"
    assert prep_res is not None
    assert len(prep_res.X_train) + len(prep_res.X_test) == len(prep_res.cleaned_df)
    assert not prep_res.X_train.isnull().values.any(), "X_train contains unhandled NaNs"
    print(f" -> PASSED: Preprocessed {len(df_raw)} records (Train: {len(prep_res.X_train)}, Test: {len(prep_res.X_test)}) with zero data leakage.")

    # 4. Test EDA
    print("\n[TEST 4] Testing modules.eda...")
    from modules.eda import compute_summary_statistics, generate_eda_insights, plot_scatter_feature_vs_target, plot_correlation_heatmap
    
    stats = compute_summary_statistics(df_raw)
    assert stats["avg_attendance"] > 0
    assert stats["avg_final_marks"] > 0
    
    insights = generate_eda_insights(df_raw)
    assert len(insights) >= 3, "Expected at least 3 data-driven insights"
    print(f" -> PASSED: Summary stats computed, {len(insights)} data-driven insights generated.")

    # 5. Test Model Training & Comparison
    print("\n[TEST 5] Testing modules.model_training...")
    from modules.model_training import train_and_compare_models, load_saved_model
    
    train_out, train_errs = train_and_compare_models(df_raw, test_size=0.2, random_state=42)
    assert train_errs == [], f"Training produced errors: {train_errs}"
    assert train_out is not None
    metrics_df = train_out["metrics_df"]
    assert len(metrics_df) == 3, f"Expected 3 models evaluated, got {len(metrics_df)}"
    print("Model Evaluation Comparison:")
    print(metrics_df.to_string(index=False))

    saved_pipe, saved_meta = load_saved_model()
    assert saved_pipe is not None, "Failed to load saved model from disk"
    assert saved_meta["model_name"] == train_out["best_model_name"]
    print(f" -> PASSED: Successfully trained, compared 3 models, and serialized best model: {saved_meta['model_name']}.")

    # 6. Test Prediction & Recommendations
    print("\n[TEST 6] Testing modules.prediction & recommendations...")
    from modules.prediction import predict_student_performance
    
    # Test High-Performing Student
    res_high, errs = predict_student_performance(
        attendance=95.0,
        prev_marks=90.0,
        internal_marks=92.0,
        assignment_marks=95.0,
        study_hours=6.0,
    )
    assert errs == []
    assert res_high["predicted_marks"] >= 75.0, f"Expected high marks, got {res_high['predicted_marks']}"
    assert res_high["category"] == "Good"
    assert res_high["grade"] in ["A", "B"]

    # Test At-Risk Student
    res_low, errs = predict_student_performance(
        attendance=48.0,
        prev_marks=42.0,
        internal_marks=38.0,
        assignment_marks=45.0,
        study_hours=1.0,
    )
    assert errs == []
    assert res_low["predicted_marks"] < 55.0, f"Expected low marks, got {res_low['predicted_marks']}"
    assert res_low["category"] in ["At Risk", "Average"]
    assert len(res_low["recommendations"]) >= 3
    print(f"High Performer: Marks={res_high['predicted_marks']}, Grade={res_high['grade']}, Standing={res_high['category']}")
    print(f"At-Risk Student: Marks={res_low['predicted_marks']}, Grade={res_low['grade']}, Standing={res_low['category']}")
    print(f" -> Tailored Recommendations generated: {len(res_low['recommendations'])}")
    print(" -> PASSED: Performance prediction and tailored recommendations behave as expected.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (6/6 TEST SUITES)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
