# frontend/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from insight_utils import show_pay_gap, show_state_diversity

st.set_page_config(page_title="DEI Analyzer", layout="wide")
st.title("📊 DEI Analyzer Dashboard")

# Load core data
@st.cache_data
def load_data():
    df_main = pd.read_csv("../data/cleaned_hr_data.csv")
    dei_score = pd.read_csv("../reports/dei_score.csv")
    feedback = pd.read_csv("../reports/feedback_analysis.csv")
    return df_main, dei_score, feedback

df, dei_df, feedback_df = load_data()

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

st.subheader("🔍 Filtered Employee Data")
st.dataframe(filtered_df)

# 📈 Diversity Charts
st.subheader("📊 Gender & State Distribution (Filtered)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Gender Breakdown**")
    st.bar_chart(filtered_df['gender'].value_counts())

with col2:
    st.markdown("**State of Origin**")
    st.bar_chart(filtered_df['state_of_origin'].value_counts())

# 🔢 Show DEI Score
st.subheader("🎯 DEI Score")
st.dataframe(dei_df)

# 💬 Feedback Summaries
st.subheader("💬 AI-based Feedback Analysis")
st.dataframe(feedback_df[['employee_id', 'sentiment', 'summary', 'inclusion_score']])

# ⬇️ Download Option
st.subheader("📥 Download Reports")
col3, col4 = st.columns(2)

with col3:
    st.download_button("Download DEI Score CSV", dei_df.to_csv(index=False), file_name="dei_score.csv")

with col4:
    st.download_button("Download Feedback Analysis CSV", feedback_df.to_csv(index=False), file_name="feedback_analysis.csv")


st.set_page_config(page_title="DEI Analyzer", layout="wide")
st.title("📊 DEI Analyzer Dashboard")

df = pd.read_csv("../data/cleaned_hr_data.csv")

with st.expander("📄 Raw Data"):
    st.dataframe(df)

# Visual Insights
show_pay_gap(df)
show_state_diversity(df)