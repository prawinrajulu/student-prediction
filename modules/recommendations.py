"""
Academic Recommendation Engine.
Provides rule-based, constructive recommendations based on student input metrics.
Includes explicit academic disclaimer.
"""
from typing import List, Dict, Any


def generate_academic_recommendations(
    attendance: float,
    prev_marks: float,
    internal_marks: float,
    assignment_marks: float,
    study_hours: float,
    predicted_marks: float,
) -> List[Dict[str, str]]:
    """
    Evaluates individual student metrics and returns tailored action points.
    All recommendations are deterministic, constructive, and responsive to user inputs.
    """
    recommendations: List[Dict[str, str]] = []

    # 1. Attendance Evaluation
    if attendance < 75.0:
        recommendations.append({
            "category": "Attendance & Classroom Engagement",
            "priority": "High" if attendance < 60.0 else "Medium",
            "icon": "⚠️",
            "suggestion": (
                f"Current attendance is {attendance:.1f}% (below the standard 75% threshold). "
                "Prioritize attending scheduled lectures consistently to prevent missing fundamental syllabus concepts and participation credits."
            ),
        })
    elif attendance >= 90.0:
        recommendations.append({
            "category": "Class Attendance",
            "priority": "Positive",
            "icon": "✅",
            "suggestion": (
                f"Excellent attendance record ({attendance:.1f}%). Continue this diligence as active class presence correlates strongly with concept retention."
            ),
        })

    # 2. Internal Assessment Evaluation
    if internal_marks < 50.0:
        recommendations.append({
            "category": "Internal Assessments & Core Concepts",
            "priority": "High",
            "icon": "⚠️",
            "suggestion": (
                f"Internal score is {internal_marks:.1f}/100. Schedule a consultation with your course instructor to review quiz feedback, "
                "clarify misconceptions early, and complete weekly practice problem sets."
            ),
        })
    elif internal_marks < 70.0:
        recommendations.append({
            "category": "Internal Assessments",
            "priority": "Medium",
            "icon": "ℹ️",
            "suggestion": (
                f"Internal score is {internal_marks:.1f}/100. Target a 10-15% improvement in upcoming unit tests through mock questions and peer study sessions."
            ),
        })

    # 3. Assignment Marks Evaluation
    if assignment_marks < 60.0:
        recommendations.append({
            "category": "Continuous Coursework & Assignments",
            "priority": "Medium",
            "icon": "📝",
            "suggestion": (
                f"Assignment score is {assignment_marks:.1f}/100. Ensure assignment deadlines are planned in advance. Focus on thorough working steps and lab documentation."
            ),
        })

    # 4. Study Hours Evaluation
    if study_hours < 2.0:
        recommendations.append({
            "category": "Daily Study Routine",
            "priority": "High",
            "icon": "⏰",
            "suggestion": (
                f"Current study time is {study_hours:.1f} hours/day. Increase dedicated self-study to at least 2.5–3.5 hours daily using structured time blocks (e.g., 45-minute focused intervals)."
            ),
        })
    elif study_hours > 7.0:
        recommendations.append({
            "category": "Study Efficiency & Burnout Prevention",
            "priority": "Low",
            "icon": "💡",
            "suggestion": (
                f"You report {study_hours:.1f} hours/day of study. Ensure your sessions focus on active recall and problem-solving rather than passive reading, and schedule adequate rest."
            ),
        })
    else:
        recommendations.append({
            "category": "Study Schedule",
            "priority": "Positive",
            "icon": "👍",
            "suggestion": (
                f"Reported {study_hours:.1f} hours/day of study is healthy. Maintain consistency and review high-weight exam topics regularly."
            ),
        })

    # 5. Foundational Preparation (Previous Marks)
    if prev_marks < 55.0:
        recommendations.append({
            "category": "Foundational Revision",
            "priority": "Medium",
            "icon": "📚",
            "suggestion": (
                f"Previous semester marks were {prev_marks:.1f}/100. Dedicate 30 minutes each week to revise prerequisite foundational concepts relevant to the current curriculum."
            ),
        })

    # 6. Overall Excellence Praise if all indicators are strong
    if (attendance >= 80.0 and internal_marks >= 75.0 and assignment_marks >= 80.0 and study_hours >= 3.0):
        recommendations.insert(0, {
            "category": "Overall Performance Profile",
            "priority": "Commendable",
            "icon": "🌟",
            "suggestion": (
                "Strong overall indicators across attendance, coursework, and daily study habits! "
                "Maintain this momentum, attempt previous year university question papers, and consider peer-tutoring."
            ),
        })

    return recommendations
