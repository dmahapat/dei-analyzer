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


def generate_ai_suggestions(df):
    suggestions = []

    grouped = df.groupby('department')
    for dept, group in grouped:
        males = group[group['gender'] == 'Male']['salary']
        females = group[group['gender'] == 'Female']['salary']

        avg_male = males.mean() if not males.empty else 0
        avg_female = females.mean() if not females.empty else 0
        gap = abs(avg_male - avg_female)

        suggestion = "✅ Pay gap is minimal."
        if gap > 100000:  # adjust based on currency (₹1L)
            suggestion = (
                "⚠️ High pay gap detected. Consider salary review or mentorship programs."
            )

        suggestions.append({
            "department": dept,
            "avg_male_salary": round(avg_male),
            "avg_female_salary": round(avg_female),
            "gap": round(gap),
            "suggestion": suggestion
        })

    return pd.DataFrame(suggestions)
