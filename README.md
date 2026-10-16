# Student Performance Prediction System Using Machine Learning

An academic machine learning project and decision-support web application designed to forecast student academic performance, identify at-risk students, and deliver constructive academic recommendations.

Developed using **Python**, **Streamlit**, and **Scikit-learn**, this system is tailored for college mini-project demonstrations and academic reviews.

---

## 1. Project Introduction

In academic environments, timely feedback is essential for student success. The **Student Performance Prediction System** utilizes basic machine learning algorithms to model the relationship between key continuous learning metrics (attendance percentage, previous semester performance, internal test marks, assignment scores, and daily study hours) and final examination outcomes.

The platform provides a modern, intuitive graphical dashboard allowing educators, administrators, and students to simulate outcomes, compare predictive algorithms, inspect data distributions, and take early corrective interventions.

---

## 2. Problem Statement

Traditionally, student academic difficulties are detected only after semester examination results are released. By that time:
* Remedial instruction and academic tutoring can no longer alter the student's semester grade.
* Students who struggle in continuous evaluations may fall below the passing threshold unnoticed.
* Educators lack a unified data-driven tool to monitor cohort trends and evaluate the impact of study habits or attendance.

This project addresses these challenges by offering an early-warning predictive system capable of identifying academic risk ahead of final examinations.

---

## 3. Project Objectives

1. **Predict Expected Examination Marks:** Forecast final examination marks (out of 100) based on student input metrics.
2. **Assign Academic Grades:** Compute letter grades (A, B, C, D, E, F) using institutional thresholds.
3. **Classify Academic Standing:** Categorize students into **Good**, **Average**, or **At Risk** status.
4. **Identify Influencing Factors:** Explain which academic habits contributed positively or negatively to the prediction.
5. **Provide Tailored Recommendations:** Deliver constructive, rule-based academic guidance tailored to individual metrics.
6. **Benchmark ML Algorithms:** Compare **Linear Regression**, **Decision Tree**, and **Random Forest** regressors using standard metrics (MAE, RMSE, R²).

---

## 4. Technologies Used

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core language for backend and analytics |
| **Frontend Framework** | Streamlit | Interactive web application and dashboard |
| **Data Manipulation** | Pandas & NumPy | Tabular data processing and numerical arrays |
| **Machine Learning** | Scikit-learn | Data preprocessing pipelines, regression models, metrics |
| **Data Visualization** | Matplotlib & Seaborn | Scientific charts, regression plots, and correlation heatmaps |
| **Model Persistence** | Joblib | Serialization of trained ML pipelines and metadata |
| **Dataset Storage** | CSV (Comma-Separated Values) | Local storage for student records and training data |

---

## 5. Explanation of All Five Modules

### Module 1: Student Data Collection (`modules/data_collection.py`)
* Interactive student profile registration form.
* Numerical validation: enforces physical ranges for attendance (0–100%), previous marks (0–100), internal marks (0–100), assignment marks (0–100), and daily study hours (0–24).
* Safe persistence to `data/student_data.csv`.
* Real-time search by Student ID or Name.
* CSV export and deletion with confirmation.
* Student names and IDs are strictly treated as metadata and never used as predictive ML features.

### Module 2: Data Preprocessing (`modules/preprocessing.py`)
* Systematic dataset loading and schema validation.
* Missing value imputation using **Median Imputation**, fit strictly on training splits to **prevent data leakage**.
* Duplicate record elimination and target variable boundary clamping `[0, 100]`.
* Dataset volume and target variance validation.
* Preprocessing summary reporting (records loaded, duplicates removed, split sizes).

### Module 3: Exploratory Data Analysis (`modules/eda.py`)
* Visual dashboard presenting 7 scientific visualizations:
  1. Attendance vs Final Examination Marks (Scatter Plot + Regression Trendline)
  2. Study Hours vs Final Examination Marks (Scatter Plot + Regression Trendline)
  3. Previous Semester Marks vs Final Examination Marks (Scatter Plot + Regression Trendline)
  4. Internal Assessment Marks vs Final Examination Marks (Scatter Plot + Regression Trendline)
  5. Academic Features Correlation Heatmap
  6. Final Examination Marks Distribution (Histogram + KDE curve)
  7. Average Final Marks by Academic Category (Bar Chart)
* Automated data-driven insights computed from actual Pearson correlation coefficients.

### Module 4: Machine Learning Model Training (`modules/model_training.py`)
* Trains and compares three algorithms on identical train-test splits (80/20):
  1. **Linear Regression** (with `StandardScaler` pipeline)
  2. **Decision Tree Regressor** (bounded depth to prevent overfitting)
  3. **Random Forest Regressor** (ensemble averaging)
* Evaluates models using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.
* Automatically selects the best model based on minimum MAE.
* Persists the selected pipeline and metadata in `models/best_student_model.joblib`.

### Module 5: Student Performance Prediction (`modules/prediction.py` & `modules/recommendations.py`)
* Real-time inference form taking attendance, prior marks, internal marks, assignments, and study hours.
* Outputs predicted marks (clamped to `[0, 100]`), letter grade, and category pill.
* Factor analysis explaining the most influential indicators.
* Rule-based academic guidance addressing attendance gaps, internal exam revision, assignment completion, and study scheduling.
* Explicit academic disclaimer clarifying predictions are statistical estimates.

---

## 6. Machine Learning Algorithms Used

1. **Linear Regression:**
   * Models the target examination mark as a weighted linear combination of input features.
   * Strengths: Highly interpretable, fast training, strong baseline for linear relationships.
   
2. **Decision Tree Regressor:**
   * Partitions feature space into orthogonal decision blocks based on mean squared error reduction.
   * Strengths: Detects non-linear threshold effects (e.g., severe mark drop below 75% attendance).

3. **Random Forest Regressor:**
   * An ensemble of diverse, randomized decision trees whose outputs are averaged.
   * Strengths: Reduces single-tree variance, captures feature interactions, robust to noise.

---

## 7. Dataset Description

The system includes a reproducible synthetic benchmark dataset stored at `data/student_performance.csv` (250 records).

### Features:
| Column Name | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | STU1001 - STU1250 | Unique student identifier |
| `attendance_percentage` | Float | 45.0 - 100.0% | Official class attendance |
| `previous_semester_marks` | Float | 35.0 - 98.0 | Aggregate marks from prior term |
| `internal_assessment_marks`| Float | 32.0 - 99.0 | Continuous internal test score |
| `assignment_marks` | Float | 40.0 - 100.0 | Coursework & assignment marks |
| `study_hours` | Float | 0.5 - 9.5 | Daily self-study hours |
| `final_exam_marks` | Float | 20.0 - 99.0 | Target: End-semester exam mark |

*Note: The dataset incorporates realistic non-linear academic variance and noise to mirror real-world classroom conditions rather than an artificial formula.*

---

## 8. Project Folder Structure

```
student_performance_prediction/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive project documentation
├── generate_data.py            # Synthetic dataset generator script
│
├── data/
│   ├── student_performance.csv # Historical training dataset (250 rows)
│   └── student_data.csv        # Student records collected via Module 1
│
├── models/
│   ├── .gitkeep                # Placeholder for model directory
│   ├── best_student_model.joblib # Saved optimal Scikit-learn model
│   └── model_metadata.joblib   # Model metrics, feature names, and metadata
│
├── modules/
│   ├── __init__.py             # Modules package initializer
│   ├── data_collection.py      # Module 1: Student data management
│   ├── preprocessing.py        # Module 2: Cleaning, imputation, train-test split
│   ├── eda.py                  # Module 3: Visualizations & automated insights
│   ├── model_training.py       # Module 4: ML model training and comparison
│   ├── prediction.py           # Module 5: Performance prediction engine
│   └── recommendations.py      # Rule-based academic recommendation generator
│
└── utils/
    ├── __init__.py             # Utils package initializer
    └── validation.py           # Input boundary & dataset schema validators
```

---

## 9. Installation Instructions

### Prerequisites
* Python 3.10 or higher installed on your computer.
* A terminal or command prompt (PowerShell / Command Prompt / Bash).

### Step 1: Open Project Directory
Navigate to the project directory:
```bash
cd "d:/student pre"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 10. How to Run the Application

Launch the Streamlit dashboard by running:
```bash
streamlit run app.py
```

Once executed, the application will automatically open in your default browser at:
`http://localhost:8501`

---

## 11. How to Upload a Custom Dataset

1. Go to the **Model Training** page via the sidebar menu.
2. Under **Dataset Selection**, choose **Upload Custom Historical CSV**.
3. Select your CSV file. Ensure your CSV contains the following column headers:
   * `student_id`
   * `attendance_percentage`
   * `previous_semester_marks`
   * `internal_assessment_marks`
   * `assignment_marks`
   * `study_hours`
   * `final_exam_marks`
4. The system validates the columns, checks for missing data, and flags any formatting issues before training.

---

## 12. How to Train the Models

1. Open the **Model Training** page.
2. Select your training dataset (Synthetic Benchmark or Custom CSV).
3. Adjust the **Testing Split Ratio** (default is 20%) and **Random Seed** (default 42).
4. Click **🚀 Train & Compare Models**.
5. The application will:
   * Clean and impute the dataset.
   * Train **Linear Regression**, **Decision Tree**, and **Random Forest**.
   * Display a comparison metrics table and an MAE bar chart.
   * Automatically serialize the best model to `models/best_student_model.joblib`.

---

## 13. How to Make Predictions

1. Open the **Student Performance Prediction** page from the sidebar.
2. Enter the student's metrics:
   * **Attendance Percentage** (e.g., `85%`)
   * **Previous Semester Marks** (e.g., `78`)
   * **Internal Assessment Marks** (e.g., `80`)
   * **Assignment Marks** (e.g., `85`)
   * **Daily Study Hours** (e.g., `4.5 hrs/day`)
3. Click **Predict Student Performance**.
4. Review the results:
   * **Predicted Final Marks** (e.g., `79.4 / 100`)
   * **Predicted Grade** (e.g., `Grade C - Good / Above Average`)
   * **Academic Standing** (e.g., `Good` / `Average` / `At Risk`)
   * **Influencing Factors** breakdown
   * **Tailored Academic Recommendations**

---

## 14. Model Evaluation Metrics

* **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
  Measures the average absolute deviation between actual and predicted marks. An MAE of 3.5 means predictions are, on average, within 3.5 marks of the actual score.
* **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
  Penalizes larger outliers more heavily than MAE.
* **R² Score (Coefficient of Determination):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  Represents the proportion of variance in final examination marks explained by the input features (scale: $-\infty$ to $1.0$).

---

## 15. Limitations of the Project

1. **Statistical Approximation:** Predictions are algorithmic estimates and cannot account for unforeseen factors such as personal illness or exam anxiety.
2. **Dataset Size Sensitivity:** Evaluation metrics on small cohorts (< 50 samples) have wider variance.
3. **Linearity Assumption:** In smaller datasets, linear models often outperform tree ensembles due to reduced risk of overfitting on modest sample sizes.
4. **Non-Cognitive Factors:** The model does not include qualitative factors like mental well-being, extracurricular commitments, or faculty teaching styles.

---

## 16. Possible Future Enhancements

* **Integration with College LMS:** Automated syncing with Canvas, Moodle, or Google Classroom.
* **Attendance Trend Analysis:** Time-series tracking of attendance drops over time rather than a single average.
* **Subject-Wise Granularity:** Multi-output regression to predict performance across individual subjects.
* **Student Portal Login:** Role-based access control for students, advisors, and department heads.
* **Automated Email Alerts:** Notification system alerting academic advisors when a student is classified as "At Risk".

---

## 17. Demonstration Checklist for Project Review

- [x] Streamlit dashboard launches without errors.
- [x] Input validation prevents negative marks and invalid percentages.
- [x] Student records can be saved, searched, downloaded, and deleted in Module 1.
- [x] Preprocessing handles missing values and avoids data leakage in Module 2.
- [x] 7 EDA visualizations and automated data-driven insights display in Module 3.
- [x] 3 ML algorithms (Linear Regression, Decision Tree, Random Forest) are trained and compared in Module 4.
- [x] Model comparison table and MAE chart display clearly.
- [x] Best model is saved and loaded via Joblib.
- [x] Prediction page outputs marks, grade, standing, factor analysis, and tailored recommendations in Module 5.
- [x] Clear academic disclaimers are visible.
