import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_pay_gap(df):
    st.subheader("💰 Average Salary by Gender")
    gender_salary = df.groupby('gender')['salary'].mean().sort_values()
    st.bar_chart(gender_salary)

    # Add insight
    gap = abs(gender_salary.iloc[0] - gender_salary.iloc[1])
    st.write(f"👉 The pay gap between genders is approximately ₹{gap:,.0f}")

    if gap > 10000:
        st.warning("⚠️ Consider investigating why such a high gap exists.")
        st.info("🛠️ Suggestion: Introduce pay audits or standardize pay bands.")

def show_state_diversity(df):
    st.subheader("🌍 Employee Count by State of Origin")
    state_counts = df['state_of_origin'].value_counts()
    st.bar_chart(state_counts)
