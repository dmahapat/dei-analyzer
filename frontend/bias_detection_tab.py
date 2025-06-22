# bias_detection_tab.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ✅ Load and preprocess data
@st.cache_data
def load_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "promotion_data.csv"))
    df = pd.read_csv(data_path)
    df["performance_score"] = pd.to_numeric(df["performance_score"], errors="coerce")
    df["tenure_years"] = pd.to_numeric(df["tenure_years"], errors="coerce")
    df.dropna(subset=["performance_score", "tenure_years", "promotion_eligible"], inplace=True)
    return df

def train_rf_model(df):
    X = df[["performance_score", "tenure_years"]]
    y = df["promotion_eligible"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X


def display_shap_chart(model, X):
    st.markdown("### 🔍 SHAP Feature Impact (Top 5)")

    try:
        # Create SHAP explainer and values
        explainer = shap.Explainer(model.predict, X)  # Safe form for sklearn
        shap_values = explainer(X)

        # Plot setup
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.bar(shap_values, max_display=5, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ SHAP plot failed: {e}")
        st.code(f"{type(e).__name__}: {str(e)}")
        st.warning("This can happen if the model input shape is off or SHAP is passed invalid data.")


def show_bias_detection():
    st.title("🏆 Promotion Bias Detector")
    st.markdown("---")
    df = load_data()
    model, X = train_rf_model(df)
    display_shap_chart(model, X)
    show_ai_insights_from_model(df)

    st.markdown("### 📄 Sample Training Data")
    st.dataframe(df.head(20))

def show_ai_insights_from_model(df):
    st.markdown("## 🧠 AI Insights: Key Drivers Behind Promotion Decisions")

    # Generate summary statistics (these can be dynamically calculated)
    perf_promoted = df[df["promotion_eligible"] == 1]["performance_score"].mean()
    perf_not_promoted = df[df["promotion_eligible"] == 0]["performance_score"].mean()
    perf_factor = round(perf_promoted / perf_not_promoted, 2) if perf_not_promoted else "N/A"

    high_tenure = df[df["tenure_years"] > 3]
    low_tenure = df[df["tenure_years"] <= 3]
    tenure_diff = round(
        (high_tenure["promotion_eligible"].mean() - low_tenure["promotion_eligible"].mean()) * 100,
        2,
    )

    insight_data = {
        "Insight": ["💼 Performance Score", "🕒 Tenure in Years"],
        "Explanation": [
            f"Employees with higher performance scores were {perf_factor}x more likely to be promoted.",
            f"Employees with more than 3 years of experience had a {tenure_diff}% higher promotion rate.",
        ],
    }

    insight_df = pd.DataFrame(insight_data)

    st.markdown("### 🔎 Auto-Interpreted Insights")
    st.table(insight_df)

