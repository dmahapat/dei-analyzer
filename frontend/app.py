# frontend/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from insight_utils import show_pay_gap, show_state_diversity, generate_ai_suggestions
from bias_detection_tab import show_bias_detection
from state_cluster_tab import show_state_clusters

# ✅ Initial Setup
st.set_page_config(page_title="DEI Analyzer", layout="wide")
st.sidebar.title("🧭 DEI Insights Navigation")

# ✅ Load Data
@st.cache_data
def load_data():
    df_main = pd.read_csv("../data/cleaned_hr_data.csv")
    dei_score = pd.read_csv("../reports/dei_score.csv")
    feedback = pd.read_csv("../reports/feedback_analysis.csv")
    promotion = pd.read_csv("../data/promotion_data.csv")
    return df_main, dei_score, feedback, promotion

df, dei_df, feedback_df, promotion_df = load_data()

# ✅ Navigation Tabs
tab = st.sidebar.radio("Choose a section", [
    "📊 DEI Insights Dashboard",
    "🤖 AI-Based DEI Suggestions",
    "🏆 Promotion Bias Detector",
    "📍 Regional Diversity Clusters"
])

# ============================================================
# 📊 TAB 1: DEI Insights Dashboard
# ============================================================
if tab == "📊 DEI Insights Dashboard":
    st.title("📊 DEI Insights Dashboard")

    # 🎯 Filters
    with st.sidebar:
        st.header("📂 Filter Data")
        depts = df['department'].unique()
        selected_dept = st.selectbox("Select Department", options=["All"] + list(depts))
        job_levels = df['job_level'].unique()
        selected_level = st.selectbox("Select Job Level", options=["All"] + list(job_levels))

    # Apply Filters
    filtered_df = df.copy()
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df['department'] == selected_dept]
    if selected_level != "All":
        filtered_df = filtered_df[filtered_df['job_level'] == selected_level]

    # 🔍 Filtered Employee Table
    st.subheader("🔍 Filtered Employee Data")
    st.dataframe(filtered_df)

    # 📈 Diversity Visuals
    st.subheader("📊 Gender & State Distribution")
    col1, col2 = st.columns(2)

    with col1:
        gender_counts = filtered_df['gender'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        state_counts = filtered_df['state_of_origin'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(state_counts.index, state_counts.values)
        plt.xticks(rotation=90)
        st.pyplot(fig2)

    # 🎯 DEI Score
    st.subheader("🎯 DEI Score")
    st.dataframe(dei_df)

    # 💬 Feedback Analysis
    st.subheader("💬 Feedback Summary")
    st.dataframe(feedback_df[['employee_id', 'sentiment', 'summary', 'inclusion_score']])

    # 📥 Download Options
    st.subheader("📥 Download Reports")
    col3, col4 = st.columns(2)
    with col3:
        st.download_button("Download DEI Score CSV", dei_df.to_csv(index=False), file_name="dei_score.csv")
    with col4:
        st.download_button("Download Feedback Analysis CSV", feedback_df.to_csv(index=False), file_name="feedback_analysis.csv")

# ============================================================
# 🤖 TAB 2: AI-Based DEI Suggestions
# ============================================================
elif tab == "🤖 AI-Based DEI Suggestions":
    st.title("🧠 AI-Based Suggestions for DEI Improvements")
    suggestions_df = generate_ai_suggestions(df)
    st.dataframe(suggestions_df)

# ============================================================
# 🏆 TAB 3: Promotion Bias Detection
# ============================================================
elif tab == "🏆 Promotion Bias Detector":
    show_bias_detection()

# ============================================================
# 📍 TAB 4: Regional Diversity Clusters
# ============================================================

elif tab == "📍 Regional Diversity Clusters":
    show_state_clusters()
