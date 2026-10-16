"""
Student Performance Prediction System Using Machine Learning
A complete, functional, and professional Streamlit application.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
from utils.validation import (
    validate_student_record,
    validate_prediction_inputs,
    validate_csv_dataset,
    REQUIRED_TRAINING_COLUMNS,
    REQUIRED_PREDICTION_FEATURES,
)
from modules.data_collection import (
    load_student_records,
    add_student_record,
    delete_student_record,
    search_student_records,
    DATA_FILE,
)
from modules.preprocessing import (
    load_dataset,
    preprocess_data,
    inspect_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from modules.eda import (
    compute_summary_statistics,
    generate_eda_insights,
    plot_scatter_feature_vs_target,
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_category_bar_chart,
)
from modules.model_training import (
    train_and_compare_models,
    load_saved_model,
    plot_model_comparison,
    MODEL_FILE,
    METADATA_FILE,
)
from modules.prediction import (
    predict_student_performance,
    calculate_grade,
    calculate_category,
    DEFAULT_GRADE_THRESHOLDS,
)

# ---------------------------------------------------------
# Page Configuration & Light Professional Academic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic, clean, blue & slate grey styling
st.markdown("""
<style>
    /* Clean background and fonts */
    .main {
        background-color: #f8fafc;
    }
    
    /* Academic Header banner */
    .academic-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff;
        padding: 20px 24px;
        border-radius: 8px;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    .academic-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
    }
    .academic-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #e2e8f0;
    }

    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Status Pills */
    .pill-good {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    .pill-average {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    .pill-risk {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }

    /* Recommendation item */
    .rec-card {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
st.sidebar.title("Student Performance System")
st.sidebar.markdown("**Machine Learning Decision Support**")
st.sidebar.markdown("---")

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Dashboard",
        "Student Data Collection",
        "Data Preprocessing",
        "Exploratory Data Analysis",
        "Model Training",
        "Student Performance Prediction",
        "About Project",
    ],
    index=0,
)

st.sidebar.markdown("---")
# Quick status widget in sidebar
pipeline, metadata = load_saved_model()
if pipeline is not None and metadata is not None:
    st.sidebar.success(f"Active ML Model: **{metadata.get('model_name', 'Trained')}**")
    st.sidebar.caption(f"Trained MAE: {metadata.get('metrics', {}).get('mae', 'N/A')} marks")
else:
    st.sidebar.info("Model Status: Baseline Ready (Train in Module 4)")

st.sidebar.caption("v1.0.0 | Academic Project Edition")


# Helper to get the primary historical training dataset
@st.cache_data
def get_benchmark_dataset() -> pd.DataFrame:
    csv_path = os.path.join("data", "student_performance.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# =========================================================
# PAGE 1: DASHBOARD
# =========================================================
if nav_page == "Dashboard":
    st.markdown("""
    <div class="academic-header">
        <h1>Academic Performance Prediction Dashboard</h1>
        <p>A machine learning system for estimating final exam performance, identifying at-risk students, and providing early academic interventions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load datasets for dashboard overview
    student_records = load_student_records()
    benchmark_df = get_benchmark_dataset()

    # Model status
    active_model_name = metadata.get("model_name", "None (Baseline)") if metadata else "Baseline (Ready to Train)"
    model_mae = f"{metadata.get('metrics', {}).get('mae', 0.0):.2f}" if metadata else "N/A"

    # Compute key stats
    total_collected_students = len(student_records)
    has_benchmark = not benchmark_df.empty

    avg_att = f"{benchmark_df['attendance_percentage'].mean():.1f}%" if has_benchmark else "N/A"
    avg_internal = f"{benchmark_df['internal_assessment_marks'].mean():.1f}" if has_benchmark else "N/A"
    avg_final = f"{benchmark_df['final_exam_marks'].mean():.1f}" if has_benchmark else "N/A"

    # Summary Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Collected Students</div>
            <div class="metric-value">{total_collected_students}</div>
            <div class="metric-sub">Module 1 Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Samples</div>
            <div class="metric-value">{len(benchmark_df)}</div>
            <div class="metric-sub">Benchmark Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Attendance</div>
            <div class="metric-value">{avg_att}</div>
            <div class="metric-sub">Cohort Average</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Final Marks</div>
            <div class="metric-value">{avg_final}</div>
            <div class="metric-sub">Out of 100</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Model</div>
            <div class="metric-value" style="font-size: 16px; margin-top:8px;">{active_model_name}</div>
            <div class="metric-sub">MAE: {model_mae} marks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Performance Overview & Cohort Health")

    if has_benchmark:
        col_chart, col_recent = st.columns([1.1, 1.3])
        with col_chart:
            st.markdown("##### Distribution of Academic Standing")
            fig_dist = plot_target_distribution(benchmark_df)
            st.pyplot(fig_dist)
            st.caption("Distribution of historical student final marks with mean and median indicators.")

        with col_recent:
            st.markdown("##### Recent Student Records")
            if not student_records.empty:
                display_df = student_records.tail(6).iloc[::-1][
                    ["student_id", "student_name", "attendance_percentage", "internal_assessment_marks", "study_hours"]
                ]
                st.dataframe(
                    display_df.rename(columns={
                        "student_id": "Student ID",
                        "student_name": "Name",
                        "attendance_percentage": "Attendance %",
                        "internal_assessment_marks": "Internals",
                        "study_hours": "Study Hrs/Day",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"Showing newest student entries from '{DATA_FILE}'. Visit **Student Data Collection** to add more.")
            else:
                st.info("No newly entered student records yet. Use the **Student Data Collection** tab to register students.")
    else:
        st.warning("No benchmark dataset found in `data/student_performance.csv`. Please run data generation or upload a CSV in Model Training.")

    st.markdown("---")
    st.markdown("### How to Get Started")
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.info("**1. Collect & Manage Data**\n\nEnter new student details in **Student Data Collection** or view current registry.")
    with g_col2:
        st.info("**2. Analyze & Train Models**\n\nInspect features in **EDA** and train 3 ML algorithms (Linear Regression, Decision Tree, Random Forest) in **Model Training**.")
    with g_col3:
        st.info("**3. Predict & Recommend**\n\nForecast marks, grades, and risk category for individual students in **Student Performance Prediction**.")


# =========================================================
# PAGE 2: MODULE 1 - STUDENT DATA COLLECTION
# =========================================================
elif nav_page == "Student Data Collection":
    st.markdown("""
    <div class="academic-header">
        <h1>Module 1: Student Data Collection</h1>
        <p>Record student academic indicators, validate inputs against physical constraints, search, and manage student records.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_form, tab_records = st.tabs(["📝 Enter New Student Record", "📋 View & Search Records"])

    with tab_form:
        st.subheader("Student Academic Profile Entry Form")
        st.caption("All numerical inputs are rigorously validated. Student ID and Name are identifiers and are not used as machine learning features.")

        with st.form("student_entry_form", clear_on_submit=False):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                input_id = st.text_input("Student ID (Unique Identifier)*", placeholder="e.g., STU202501").strip()
                input_name = st.text_input("Full Name*", placeholder="e.g., John Doe").strip()
                input_att = st.number_input(
                    "Attendance Percentage (0 - 100)*",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.5,
                    help="Official classroom attendance percentage.",
                )
                input_study = st.number_input(
                    "Daily Self-Study Hours (0 - 24)*",
                    min_value=0.0,
                    max_value=24.0,
                    value=3.5,
                    step=0.5,
                    help="Typical self-study commitment per day.",
                )

            with f_col2:
                input_prev = st.number_input(
                    "Previous Semester Marks (0 - 100)*",
                    min_value=0.0,
                    max_value=100.0,
                    value=70.0,
                    step=0.5,
                    help="Aggregate marks from previous academic term.",
                )
                input_internal = st.number_input(
                    "Internal Assessment Marks (0 - 100)*",
                    min_value=0.0,
                    max_value=100.0,
                    value=72.0,
                    step=0.5,
                    help="Cumulative score from internal periodic exams/quizzes.",
                )
                input_assignment = st.number_input(
                    "Assignment Marks (0 - 100)*",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=0.5,
                    help="Continuous assessment and coursework score.",
                )

            submit_btn = st.form_submit_button("Save Student Record", use_container_width=True, type="primary")

            if submit_btn:
                # Validation
                success, msg = add_student_record(
                    student_id=input_id,
                    student_name=input_name,
                    attendance=input_att,
                    prev_marks=input_prev,
                    internal_marks=input_internal,
                    assignment_marks=input_assignment,
                    study_hours=input_study,
                )
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ Submission Failed: {msg}")

    with tab_records:
        st.subheader("Collected Student Records Registry")
        all_records = load_student_records()

        search_col, download_col = st.columns([3, 1])
        with search_col:
            search_query = st.text_input("🔍 Search by Student ID or Name", placeholder="Type ID or Name...")
        with download_col:
            if not all_records.empty:
                csv_bytes = all_records.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_bytes,
                    file_name="student_records_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        filtered_df = search_student_records(search_query) if search_query else all_records

        if not filtered_df.empty:
            st.dataframe(
                filtered_df.rename(columns={
                    "student_id": "Student ID",
                    "student_name": "Student Name",
                    "attendance_percentage": "Attendance (%)",
                    "previous_semester_marks": "Prev Sem Marks",
                    "internal_assessment_marks": "Internal Marks",
                    "assignment_marks": "Assignment Marks",
                    "study_hours": "Study Hrs/Day",
                    "submitted_at": "Timestamp",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Showing {len(filtered_df)} of {len(all_records)} total recorded students.")

            # Record Deletion Section with Explicit Confirmation
            st.markdown("---")
            st.markdown("##### 🗑️ Delete Student Record")
            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                student_ids_list = all_records["student_id"].tolist()
                target_del_id = st.selectbox("Select Student ID to Remove", options=student_ids_list)
            with del_col2:
                st.write("")
                st.write("")
                confirm_del = st.checkbox("Confirm Deletion")
                if st.button("Delete Record", type="secondary", disabled=not confirm_del, use_container_width=True):
                    del_ok, del_msg = delete_student_record(target_del_id)
                    if del_ok:
                        st.success(del_msg)
                        st.rerun()
                    else:
                        st.error(del_msg)
        else:
            st.info("No student records found matching the query.")


# =========================================================
# PAGE 3: MODULE 2 - DATA PREPROCESSING
# =========================================================
elif nav_page == "Data Preprocessing":
    st.markdown("""
    <div class="academic-header">
        <h1>Module 2: Data Preprocessing & Validation</h1>
        <p>Systematic data cleaning, missing value imputation, duplicate elimination, range validation, and strict train-test partitioning to prevent data leakage.</p>
    </div>
    """, unsafe_allow_html=True)

    # Source Selection
    st.subheader("Select Dataset for Preprocessing")
    dataset_source = st.radio(
        "Choose Data Source:",
        ["Benchmark Synthetic Dataset (student_performance.csv)", "Upload Custom CSV File"],
        horizontal=True,
    )

    df_raw = None
    if dataset_source.startswith("Benchmark"):
        df_raw = get_benchmark_dataset()
        st.info("Using preloaded academic benchmark dataset (`data/student_performance.csv`).")
    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = load_dataset(uploaded_file)
                is_valid, v_errs, v_warns = validate_csv_dataset(df_raw)
                if not is_valid:
                    st.error("Uploaded CSV failed schema validation:")
                    for e in v_errs:
                        st.error(f"• {e}")
                    df_raw = None
                else:
                    for w in v_warns:
                        st.warning(f"• {w}")
            except Exception as e:
                st.error(f"Error parsing CSV: {str(e)}")
                df_raw = None

    if df_raw is not None and not df_raw.empty:
        # Preprocessing settings
        st.markdown("### Preprocessing Configuration")
        prep_c1, prep_c2, prep_c3 = st.columns(3)
        with prep_c1:
            test_split = st.slider("Testing Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        with prep_c2:
            seed_val = st.number_input("Reproducibility Seed (random_state)", min_value=0, max_value=999, value=42)
        with prep_c3:
            scale_preview = st.checkbox("StandardScaler Feature Scaling", value=False, help="Applied automatically inside ML pipelines where required.")

        # Run Preprocessing
        prep_result, prep_errors = preprocess_data(
            df=df_raw,
            test_size=test_split,
            random_state=seed_val,
            apply_scaling=scale_preview,
        )

        if prep_errors:
            for err in prep_errors:
                st.error(f"Preprocessing Error: {err}")
        elif prep_result is not None:
            summary = prep_result.summary

            # Preprocessing Summary Cards
            st.markdown("### Preprocessing Pipeline Summary")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Total Records Loaded", summary["total_records_loaded"])
            s2.metric("Missing Values Imputed", summary["missing_values_detected"])
            s3.metric("Duplicates Removed", summary["duplicates_removed"])
            s4.metric("Clean Records Available", summary["records_available_after_cleaning"])
            s5.metric("Train / Test Split", f"{summary['train_set_size']} / {summary['test_set_size']}")

            # Data Leakage Prevention Notice
            st.success("🔒 **Data Leakage Safeguard:** Missing values were imputed strictly using training statistics (`X_train` median), ensuring no test set information leaked into model inputs.")

            # Columns & Targets
            st.markdown("#### Selected Features & Target Definition")
            f_col, t_col = st.columns(2)
            with f_col:
                st.markdown("**Input Features (X):**")
                for f in summary["selected_features"]:
                    st.write(f"- `{f}` ({f.replace('_', ' ').title()})")
            with t_col:
                st.markdown("**Target Variable (y):**")
                st.write(f"- `{summary['target_variable']}` (Final Examination Marks out of 100)")
                st.write(f"- **Imputation Strategy:** `{summary['imputation_strategy']}`")

            # Clean Data Preview
            st.markdown("#### Cleaned Dataset Preview (First 8 Rows)")
            st.dataframe(prep_result.cleaned_df.head(8), use_container_width=True)

    else:
        st.info("Please select or upload a valid dataset to view preprocessing details.")


# =========================================================
# PAGE 4: MODULE 3 - EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
elif nav_page == "Exploratory Data Analysis":
    st.markdown("""
    <div class="academic-header">
        <h1>Module 3: Exploratory Data Analysis (EDA)</h1>
        <p>Interactive visual exploration of student performance drivers, correlation matrices, score distributions, and data-driven insights.</p>
    </div>
    """, unsafe_allow_html=True)

    df_eda = get_benchmark_dataset()

    if df_eda.empty:
        st.warning("No dataset available for EDA. Please verify `data/student_performance.csv` exists.")
    else:
        # Summary Statistics
        stats = compute_summary_statistics(df_eda)
        st.markdown("### Academic Summary Statistics")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Cohort", f"{stats['total_records']} students")
        c2.metric("Avg Attendance", f"{stats['avg_attendance']}%")
        c3.metric("Avg Study Hours", f"{stats['avg_study_hours']} hrs/day")
        c4.metric("Avg Internal Marks", f"{stats['avg_internal_marks']}/100")
        c5.metric("Avg Final Marks", f"{stats['avg_final_marks']}/100")
        c6.metric("Final Range", f"{stats['lowest_final_marks']} - {stats['highest_final_marks']}")

        st.markdown("---")

        # Automated EDA Insights from Actual Dataset
        st.markdown("### 🧠 Automated Data-Driven Insights")
        st.caption("These conclusions are computed mathematically from Pearson correlation coefficients and distribution analysis of the actual data:")
        insights = generate_eda_insights(df_eda)

        i_col1, i_col2 = st.columns(2)
        for idx, ins in enumerate(insights):
            target_col = i_col1 if idx % 2 == 0 else i_col2
            with target_col:
                st.markdown(f"""
                <div class="rec-card">
                    <strong>{ins['title']}</strong><br/>
                    <span style="font-size: 13.5px; color: #334155;">{ins['detail']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Visualizations Dashboard")

        # Visualizations 1 & 2: Scatter plots
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("##### 1. Attendance vs Final Examination Marks")
            fig1 = plot_scatter_feature_vs_target(
                df_eda,
                "attendance_percentage",
                "Attendance Percentage (%)",
                color="#1d4ed8",
            )
            st.pyplot(fig1)

        with v2:
            st.markdown("##### 2. Study Hours vs Final Examination Marks")
            fig2 = plot_scatter_feature_vs_target(
                df_eda,
                "study_hours",
                "Daily Study Hours",
                color="#0f766e",
            )
            st.pyplot(fig2)

        # Visualizations 3 & 4: Previous semester & internal assessment scatter plots
        v3, v4 = st.columns(2)
        with v3:
            st.markdown("##### 3. Previous Semester Marks vs Final Marks")
            fig3 = plot_scatter_feature_vs_target(
                df_eda,
                "previous_semester_marks",
                "Previous Semester Marks",
                color="#6d28d9",
            )
            st.pyplot(fig3)

        with v4:
            st.markdown("##### 4. Internal Assessment Marks vs Final Marks")
            fig4 = plot_scatter_feature_vs_target(
                df_eda,
                "internal_assessment_marks",
                "Internal Assessment Marks",
                color="#b45309",
            )
            st.pyplot(fig4)

        # Visualizations 5 & 6: Correlation Heatmap and Distribution
        v5, v6 = st.columns([1.1, 1.0])
        with v5:
            st.markdown("##### 5. Academic Features Correlation Heatmap")
            fig5 = plot_correlation_heatmap(df_eda)
            st.pyplot(fig5)

        with v6:
            st.markdown("##### 6. Distribution of Final Examination Marks")
            fig6 = plot_target_distribution(df_eda)
            st.pyplot(fig6)

        # Visualization 7: Performance Category Bar Chart
        st.markdown("##### 7. Average Final Marks by Academic Category")
        fig7 = plot_category_bar_chart(df_eda)
        st.pyplot(fig7)


# =========================================================
# PAGE 5: MODULE 4 - MACHINE LEARNING MODEL TRAINING
# =========================================================
elif nav_page == "Model Training":
    st.markdown("""
    <div class="academic-header">
        <h1>Module 4: Machine Learning Model Training & Comparison</h1>
        <p>Train and rigorously evaluate Linear Regression, Decision Tree, and Random Forest Regressors using standardized metrics (MAE, RMSE, R²).</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1. Training Dataset Selection")
    train_source = st.radio(
        "Select Dataset for Model Training:",
        ["Synthetic Benchmark Dataset (`student_performance.csv`)", "Upload Custom Historical CSV"],
        horizontal=True,
    )

    df_train = None
    if train_source.startswith("Synthetic"):
        df_train = get_benchmark_dataset()
        st.info("Loaded benchmark dataset (250 student samples with realistic noise).")
    else:
        uploaded_train = st.file_uploader("Upload Historical Training CSV", type=["csv"], key="train_uploader")
        if uploaded_train is not None:
            try:
                df_train = load_dataset(uploaded_train)
                is_val, v_err, v_warn = validate_csv_dataset(df_train)
                if not is_val:
                    st.error("Uploaded CSV cannot be used for training:")
                    for e in v_err:
                        st.error(f"• {e}")
                    df_train = None
                else:
                    for w in v_warn:
                        st.warning(f"• {w}")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                df_train = None

    if df_train is not None and not df_train.empty:
        st.markdown("### 2. Training Parameters")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            split_pct = st.slider("Testing Dataset Size (%)", min_value=10, max_value=35, value=20, step=5)
        with p_c2:
            fixed_seed = st.number_input("Random Seed for Reproducibility", min_value=1, max_value=999, value=42)

        st.caption("Training will evaluate: **1. Linear Regression**, **2. Decision Tree Regressor**, and **3. Random Forest Regressor** under the exact same data split.")

        if st.button("🚀 Train & Compare Models", type="primary", use_container_width=True):
            with st.spinner("Training models, computing cross-evaluations, and selecting best algorithm..."):
                training_output, train_errors = train_and_compare_models(
                    df=df_train,
                    test_size=split_pct / 100.0,
                    random_state=fixed_seed,
                )

                if train_errors:
                    for err in train_errors:
                        st.error(f"Training Failed: {err}")
                elif training_output:
                    st.success("✅ Training and evaluation completed successfully!")
                    st.session_state["training_output"] = training_output

        # Display training results if stored in session or freshly computed
        if "training_output" in st.session_state:
            t_out = st.session_state["training_output"]
            metrics_df = t_out["metrics_df"]
            best_name = t_out["best_model_name"]
            best_m = t_out["best_metrics"]

            st.markdown("---")
            st.markdown("### 3. Model Evaluation & Comparison Results")

            # Best Model Banner
            st.markdown(f"""
            <div style="background-color: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
                <span style="font-size: 18px; font-weight: 700; color: #065f46;">🏆 Selected Best-Performing Model: {best_name}</span><br/>
                <span style="color: #047857; font-size: 14px;">
                    Selected based on lowest Mean Absolute Error (MAE: <strong>{best_m['mae']}</strong> marks).
                    Persisted as active model in <code>models/best_student_model.joblib</code> for real-time predictions.
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Table and Bar Chart
            col_tbl, col_chart = st.columns([1.1, 1.2])
            with col_tbl:
                st.markdown("##### Comparative Performance Metrics Table")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                st.markdown("""
                **Metric Explanations:**
                - **MAE (Mean Absolute Error):** Average magnitude of absolute prediction error in exam marks. Lower is better.
                - **RMSE (Root Mean Squared Error):** Penalizes larger outlier prediction errors. Lower is better.
                - **R² Score:** Proportion of mark variance explained by the model features (0 to 1). Higher is better.
                """)

            with col_chart:
                st.markdown("##### MAE Comparison Across Models")
                fig_comp = plot_model_comparison(metrics_df)
                st.pyplot(fig_comp)

            # Academic Reflection & Limitations
            st.markdown("---")
            st.markdown("##### 📌 Scientific Notes & Training Limitations")
            st.caption("""
            - **No Artificial Assumption:** Complex non-linear models (like Random Forests) do not always outperform Linear Regression on moderately linear academic datasets, especially when sample sizes are small. Model selection is purely empirical based on test MAE.
            - **Data Leakage Safeguards:** Feature scaling and imputer fittings were strictly constrained to the training partition (`X_train`).
            - **Small Dataset Consideration:** In small training sets (< 100 samples), evaluation metrics can exhibit higher variance. Increasing data points with genuine examination records yields more stable generalization.
            """)


# =========================================================
# PAGE 6: MODULE 5 - STUDENT PERFORMANCE PREDICTION
# =========================================================
elif nav_page == "Student Performance Prediction":
    st.markdown("""
    <div class="academic-header">
        <h1>Module 5: Student Performance Prediction</h1>
        <p>Predict expected final examination marks, letter grade, academic standing category, and constructive improvement recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check model readiness
    pipeline, metadata = load_saved_model()
    if pipeline is None:
        st.warning("⚠️ No trained model found on disk yet. You can train one in **Model Training**, or click below to initialize the baseline model using historical data.")
        if st.button("Initialize Benchmark Model Now", type="primary"):
            df_hist = get_benchmark_dataset()
            train_and_compare_models(df_hist)
            st.rerun()

    active_name = metadata.get("model_name", "Trained Scikit-Learn Model") if metadata else "Baseline Model"
    active_mae = metadata.get("metrics", {}).get("mae", "N/A") if metadata else "N/A"

    st.info(f"Using Active Model: **{active_name}** (Test MAE: ~{active_mae} marks). All predictions are estimates.")

    # Prediction Input Form
    st.subheader("Student Academic Input Parameters")
    with st.form("prediction_form"):
        p1, p2 = st.columns(2)
        with p1:
            pred_att = st.number_input("Attendance Percentage (0 - 100)*", min_value=0.0, max_value=100.0, value=82.0, step=0.5)
            pred_study = st.number_input("Daily Study Hours (0 - 24)*", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
            pred_prev = st.number_input("Previous Semester Marks (0 - 100)*", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

        with p2:
            pred_internal = st.number_input("Internal Assessment Marks (0 - 100)*", min_value=0.0, max_value=100.0, value=78.0, step=0.5)
            pred_assignment = st.number_input("Assignment Marks (0 - 100)*", min_value=0.0, max_value=100.0, value=85.0, step=0.5)

            # Optional Configurable Grading Thresholds Expander
            with st.expander("⚙️ View / Adjust Grade Thresholds (Configurable)"):
                st.caption("College institutional thresholds for grade assignment:")
                st.text("A: 90-100 | B: 80-89.99 | C: 70-79.99 | D: 60-69.99 | E: 50-59.99 | F: <50")

        predict_button = st.form_submit_button("Predict Student Performance", type="primary", use_container_width=True)

    if predict_button:
        with st.spinner("Analyzing academic indicators and computing prediction..."):
            pred_result, pred_errors = predict_student_performance(
                attendance=pred_att,
                prev_marks=pred_prev,
                internal_marks=pred_internal,
                assignment_marks=pred_assignment,
                study_hours=pred_study,
            )

            if pred_errors:
                for err in pred_errors:
                    st.error(f"Prediction Error: {err}")
            elif pred_result:
                st.markdown("---")
                st.subheader("Prediction Results & Performance Assessment")

                marks = pred_result["predicted_marks"]
                grade = pred_result["grade"]
                grade_desc = pred_result["grade_description"]
                cat = pred_result["category"]
                cat_desc = pred_result["category_description"]

                # Category Pill Style
                pill_class = "pill-good" if cat == "Good" else "pill-average" if cat == "Average" else "pill-risk"

                # Result Cards
                res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                with res_c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Final Marks</div>
                        <div class="metric-value" style="color: #1e40af;">{marks:.1f} / 100</div>
                        <div class="metric-sub">Expected Exam Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res_c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Grade</div>
                        <div class="metric-value" style="color: #0f766e;">Grade {grade}</div>
                        <div class="metric-sub">{grade_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res_c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Academic Standing</div>
                        <div style="margin-top: 8px;">
                            <span class="{pill_class}">{cat}</span>
                        </div>
                        <div class="metric-sub" style="margin-top: 8px;">{cat_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res_c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predictive Algorithm</div>
                        <div class="metric-value" style="font-size: 16px; margin-top: 6px;">{pred_result['model_name']}</div>
                        <div class="metric-sub">Trained: {pred_result['trained_at']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Prediction Factor Analysis
                st.markdown("#### 🔍 Prediction Factor Analysis")
                st.markdown(f"""
                <div style="background-color: #f1f5f9; border-left: 4px solid #475569; padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
                    <strong>Influencing Factors:</strong> {pred_result['explanation']}
                </div>
                """, unsafe_allow_html=True)

                # Constructive Rule-Based Academic Recommendations
                st.markdown("#### 💡 Tailored Academic Recommendations")
                st.caption("Actionable guidance tailored specifically to the input values provided:")

                for rec in pred_result["recommendations"]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <strong>{rec['icon']} {rec['category']} ({rec['priority']} Priority)</strong><br/>
                        <span style="font-size: 13.5px; color: #334155;">{rec['suggestion']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Academic Disclaimer
                st.markdown("""
                > **Important Academic Disclaimer:**
                > The predicted final marks and performance categories are algorithmic statistical approximations produced by historical regression patterns. They do **not** represent guaranteed examination outcomes. External factors such as exam difficulty, revision quality, and health can influence actual performance. These recommendations serve strictly as study planning guidance.
                """)


# =========================================================
# PAGE 7: ABOUT PROJECT
# =========================================================
elif nav_page == "About Project":
    st.markdown("""
    <div class="academic-header">
        <h1>About the Project</h1>
        <p>Comprehensive academic documentation, problem formulation, system methodology, and algorithm comparison for student performance prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 1. Project Overview & Problem Statement
    In contemporary higher education, identifying students who struggle academically often occurs too late — typically after semester examination results are officially published. 
    
    The **Student Performance Prediction System** provides an intelligent early-warning decision support platform. By modeling relationships between continuous academic indicators (attendance, previous academic history, internal quizzes, assignments, and study hours), faculty and students can forecast expected outcomes, diagnose weaknesses, and intervene constructively before final examinations.

    ---

    ### 2. System Architecture & Modules
    The project is structured into five modular components:

    1. **Module 1: Student Data Collection**
       - Interactive user interface to register student academic profiles.
       - Rigorous numerical boundary validation (percentages [0-100], study hours [0-24]).
       - Searchable local persistence (`data/student_data.csv`) with deletion safeguards.
       
    2. **Module 2: Data Preprocessing & Validation**
       - Detection of missing values and duplicate records.
       - Non-leaking median imputation fit strictly on training splits.
       - Verification of target variance and sample volume thresholds.

    3. **Module 3: Exploratory Data Analysis (EDA)**
       - Seven distinct visualizations (scatter plots with trendlines, correlation heatmap, distribution histograms, and category bar charts).
       - Automated analytical observations generated dynamically from actual dataset coefficients.

    4. **Module 4: Machine Learning Model Training**
       - Comparative training of **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**.
       - Standardized evaluation under identical data splits using **MAE**, **RMSE**, and **R² Score**.
       - Automated selection and Joblib persistence of the optimal regression pipeline.

    5. **Module 5: Student Performance Prediction & Advisory**
       - Real-time score inference clamped to physical exam marks [0 - 100].
       - Letter grading (A through F) and standing categorization (Good, Average, At Risk).
       - Input-specific, constructive academic study recommendations.

    ---

    ### 3. Machine Learning Algorithms Employed

    | Algorithm | Working Principle | Key Strength |
    | :--- | :--- | :--- |
    | **Linear Regression** | Models target as a linear combination of weighted feature coefficients with standard scaling. | Highly interpretable, low computational overhead, excellent baseline for linear trends. |
    | **Decision Tree Regressor** | Recursively splits feature space into rectangular decision regions based on mean squared error reduction. | Captures non-linear thresholds (e.g. sharp mark drop when attendance < 75%). |
    | **Random Forest Regressor** | Ensemble of randomized decision trees aggregating individual predictions via averaging. | Mitigates overfitting, handles subtle feature interactions, robust to minor noise. |

    ---

    ### 4. Technical Stack
    - **Language:** Python 3.10+
    - **Frontend / UI:** Streamlit
    - **Data Manipulation:** Pandas, NumPy
    - **Machine Learning:** Scikit-learn
    - **Visualization:** Matplotlib, Seaborn
    - **Model Persistence:** Joblib

    ---

    ### 5. Project Limitations & Ethical Considerations
    - **Algorithmic Estimates:** Model predictions are probabilistic estimates, not definitive prophecies.
    - **Dataset Size Dependency:** Training on small cohorts (< 100 samples) can yield wider evaluation variance.
    - **Holistic Context:** Non-quantified factors such as student health, socioeconomic background, and course difficulty also influence examination performance.
    """)

# Footer across all pages
st.markdown("""
<div class="footer-note">
    Student Performance Prediction System Using Machine Learning | College Mini-Project Demo Edition
</div>
""", unsafe_allow_html=True)
