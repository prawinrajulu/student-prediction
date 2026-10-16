"""
Module 3: Exploratory Data Analysis (EDA).
Generates publication-quality charts and automated data-driven insights.
Uses Matplotlib and Seaborn with clean academic aesthetics.
"""
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent, clean academic aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
})

# Performance categorization helper
def categorize_marks(mark: float) -> str:
    if mark < 50.0:
        return "At Risk"
    elif mark < 75.0:
        return "Average"
    else:
        return "Good"


def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes academic summary statistics for display cards."""
    final_col = "final_exam_marks" if "final_exam_marks" in df.columns else None

    stats = {
        "total_records": len(df),
        "avg_attendance": round(df["attendance_percentage"].mean(), 1) if "attendance_percentage" in df.columns else 0.0,
        "avg_study_hours": round(df["study_hours"].mean(), 1) if "study_hours" in df.columns else 0.0,
        "avg_internal_marks": round(df["internal_assessment_marks"].mean(), 1) if "internal_assessment_marks" in df.columns else 0.0,
        "avg_prev_marks": round(df["previous_semester_marks"].mean(), 1) if "previous_semester_marks" in df.columns else 0.0,
        "avg_final_marks": round(df[final_col].mean(), 1) if final_col else None,
        "highest_final_marks": round(df[final_col].max(), 1) if final_col else None,
        "lowest_final_marks": round(df[final_col].min(), 1) if final_col else None,
    }
    return stats


def generate_eda_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Computes data-driven observations from the actual dataset rather than
    assuming arbitrary relationships.
    """
    insights: List[Dict[str, str]] = []

    if "final_exam_marks" not in df.columns:
        return [{"title": "Notice", "detail": "Dataset lacks final examination marks for correlation analysis."}]

    numeric_cols = [
        "attendance_percentage",
        "previous_semester_marks",
        "internal_assessment_marks",
        "assignment_marks",
        "study_hours",
    ]

    valid_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[valid_cols + ["final_exam_marks"]].corr()
    target_corr = corr_matrix["final_exam_marks"].drop("final_exam_marks")

    # 1. Strongest Predictor
    if not target_corr.empty:
        strongest_feature = target_corr.abs().idxmax()
        strongest_val = target_corr[strongest_feature]
        strength_desc = "strong" if abs(strongest_val) >= 0.6 else "moderate" if abs(strongest_val) >= 0.35 else "mild"
        direction = "positive" if strongest_val > 0 else "negative"
        display_name = strongest_feature.replace("_", " ").title()

        insights.append({
            "title": "Primary Performance Correlate",
            "detail": f"'{display_name}' shows the strongest {direction} correlation (r = {strongest_val:.2f}) with final examination marks, indicating a {strength_desc} relationship in this cohort.",
        })

    # 2. Attendance Impact
    if "attendance_percentage" in target_corr:
        r_att = target_corr["attendance_percentage"]
        if r_att > 0.4:
            att_msg = f"Consistent class attendance demonstrates a clear positive impact (r = {r_att:.2f}) on academic performance."
        elif r_att > 0.15:
            att_msg = f"Class attendance has a moderate positive association (r = {r_att:.2f}) with examination scores."
        else:
            att_msg = f"Attendance exhibits a weak correlation (r = {r_att:.2f}) with final marks in this specific sample."
        insights.append({"title": "Attendance Pattern", "detail": att_msg})

    # 3. Study Hours Impact
    if "study_hours" in target_corr:
        r_study = target_corr["study_hours"]
        if r_study > 0.35:
            study_msg = f"Higher daily study hours are positively associated with final exam performance (r = {r_study:.2f})."
        elif r_study > 0.1:
            study_msg = f"Daily study hours show a modest positive relationship (r = {r_study:.2f}), suggesting study efficiency also matters."
        else:
            study_msg = f"Daily study hours have minimal linear correlation (r = {r_study:.2f}) with final marks in this dataset."
        insights.append({"title": "Study Duration Relationship", "detail": study_msg})

    # 4. Cohort Performance Distribution
    df_cat = df["final_exam_marks"].apply(categorize_marks)
    counts = df_cat.value_counts(normalize=True) * 100
    at_risk_pct = counts.get("At Risk", 0.0)
    good_pct = counts.get("Good", 0.0)
    avg_pct = counts.get("Average", 0.0)

    insights.append({
        "title": "Academic Health Breakdown",
        "detail": f"{good_pct:.1f}% of students achieved 'Good' standing (>=75 marks), {avg_pct:.1f}% in 'Average' (50-74.99 marks), and {at_risk_pct:.1f}% are 'At Risk' (<50 marks) needing intervention.",
    })

    return insights


# Individual plot generators returning Matplotlib figure objects

def plot_scatter_feature_vs_target(
    df: pd.DataFrame,
    feature_col: str,
    feature_label: str,
    color: str = "#1f77b4"
) -> plt.Figure:
    """Generates scatter plot with regression trendline for a single feature vs target."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    sns.regplot(
        data=df,
        x=feature_col,
        y="final_exam_marks",
        ax=ax,
        scatter_kws={"alpha": 0.65, "color": color, "s": 35},
        line_kws={"color": "#d62728", "linewidth": 2},
    )

    corr = df[[feature_col, "final_exam_marks"]].corr().iloc[0, 1]
    ax.set_title(f"{feature_label} vs Final Examination Marks\n(Pearson r = {corr:.2f})", fontsize=11, fontweight="bold")
    ax.set_xlabel(feature_label)
    ax.set_ylabel("Final Examination Marks (out of 100)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Generates correlation heatmap across all numerical academic features."""
    cols = [
        "attendance_percentage",
        "previous_semester_marks",
        "internal_assessment_marks",
        "assignment_marks",
        "study_hours",
        "final_exam_marks",
    ]
    valid_cols = [c for c in cols if c in df.columns]
    corr = df[valid_cols].corr()

    labels = [c.replace("_", " ").title() for c in valid_cols]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=-0.2,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
    )
    ax.set_title("Academic Features Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Generates histogram with KDE for final exam marks."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    sns.histplot(
        df["final_exam_marks"],
        kde=True,
        color="#2b5c8f",
        ax=ax,
        bins=15,
        edgecolor="white",
        line_kws={"linewidth": 2},
    )

    mean_val = df["final_exam_marks"].mean()
    median_val = df["final_exam_marks"].median()

    ax.axvline(mean_val, color="#e45756", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="#4c78a8", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.1f}")

    ax.set_title("Distribution of Final Examination Marks", fontsize=11, fontweight="bold")
    ax.set_xlabel("Final Examination Marks (0 - 100)")
    ax.set_ylabel("Student Count")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def plot_category_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Generates bar chart showing average final marks by performance category."""
    df_temp = df.copy()
    df_temp["category"] = df_temp["final_exam_marks"].apply(categorize_marks)

    order = ["At Risk", "Average", "Good"]
    # Group with category as index to preserve category names
    grouped = df_temp.groupby("category")["final_exam_marks"].agg(
        mean="mean",
        count="count"
    )

    # Order categories logically: At Risk -> Average -> Good
    existing_cats = [c for c in order if c in grouped.index]
    cat_stats = grouped.loc[existing_cats] if existing_cats else grouped

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"At Risk": "#e45756", "Average": "#f58518", "Good": "#54a24b"}
    bar_colors = [colors.get(c, "#4c78a8") for c in cat_stats.index]

    bars = ax.bar(cat_stats.index, cat_stats["mean"], color=bar_colors, edgecolor="black", linewidth=0.6, width=0.55)

    for bar, (_, row) in zip(bars, cat_stats.iterrows()):
        height = bar.get_height()
        count_val = int(row["count"]) if pd.notna(row.get("count")) else 0
        mean_val = float(height) if pd.notna(height) else 0.0
        ax.annotate(
            f"Avg: {mean_val:.1f}\n(n={count_val})",
            xy=(bar.get_x() + bar.get_width() / 2, mean_val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_title("Average Final Marks by Academic Category", fontsize=11, fontweight="bold")
    ax.set_xlabel("Performance Category")
    ax.set_ylabel("Average Final Examination Marks")
    fig.tight_layout()
    return fig
