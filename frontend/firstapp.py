import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import openai
import os

# Setup OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI-Powered HR Insights Dashboard", layout="wide")
st.title("🤖 AI-Powered HR Insights Dashboard")

uploaded_file = st.file_uploader("Upload Cleaned HR CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    with st.sidebar:
        st.markdown("### Filters")
        selected_dept = st.multiselect("Filter by Department", options=df['department'].unique(), default=list(df['department'].unique()))
        df = df[df['department'].isin(selected_dept)]

    tabs = st.tabs([
        "Attrition by Department",
        "Salary by Gender",
        "Salary vs Performance",
        "Tenure by Job Level",
        "Performance by Department",
        "Feedback NLP + AI",
        "Exit Status by State",
        "Salary by Department",
        "Gender-wise Exit Ratio",
        "AI Executive Summary",
        "AI Promotion Bias Explanation",
        "AI Scenario Simulation",
        "Chat with HR Data"
    ])

    with tabs[0]:
        st.subheader("Attrition by Department")
        attrition_dept = df.groupby('department')['exit_status'].value_counts(normalize=True).unstack().fillna(0) * 100

        attrition_dept = df.groupby('department')['exit_status'].value_counts(normalize=True).unstack().fillna(0) * 100

        # Ensure only numeric data is plotted
        if not attrition_dept.empty and attrition_dept.select_dtypes(include='number').shape[1] > 0:
            fig, ax = plt.subplots()
            attrition_dept.plot(kind='bar', stacked=True, ax=ax)
            ax.set_ylabel(\"% Employees\")
            ax.set_title(\"Attrition % by Department\")
            st.pyplot(fig)
        else:
            st.warning(\"No numeric attrition data available to plot.\")







        
        st.dataframe(attrition_dept.style.format("{:.2f}"))
        fig, ax = plt.subplots()
        attrition_dept.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel("% Employees")
        ax.set_title("Attrition % by Department")
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Salary Distribution by Gender")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='gender', y='salary', ax=ax)
        st.pyplot(fig)

    with tabs[2]:
        st.subheader("Salary vs Performance Correlation")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='performance_score', y='salary', hue='gender', ax=ax)
        st.pyplot(fig)
        corr = df[['salary', 'performance_score']].corr().iloc[0, 1]
        st.metric("Correlation between Salary & Performance", f"{corr:.2f}")

    with tabs[3]:
        st.subheader("Average Tenure by Job Level")
        tenure_level = df.groupby('job_level')['tenure_years'].mean().sort_values()
        fig, ax = plt.subplots()
        tenure_level.plot(kind='barh', ax=ax)
        ax.set_xlabel("Years")
        st.pyplot(fig)

    with tabs[4]:
        st.subheader("Average Performance Score by Department")
        perf_dept = df.groupby('department')['performance_score'].mean().sort_values()
        st.bar_chart(perf_dept)

    with tabs[5]:
        st.subheader("🔍 AI-Analyzed Feedback Summary")
        feedbacks = df[df['exit_status'] == 'Exited']['feedback_text'].dropna().tolist()
        combined_text = "\n".join(feedbacks[:15])

        if st.button("Analyze Feedback with AI"):
            with st.spinner("Thinking..."):
                prompt = f"""
                Analyze the following employee feedback data. Return:
                - Top 3 concerns
                - Emotional tone summary
                - Suggestions for HR

                Feedback:
                {combined_text}
                """
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"Error: {e}")

    with tabs[6]:
        st.subheader("Exit Status by State of Origin")
        state_exit = df.groupby('state_of_origin')['exit_status'].value_counts(normalize=True).unstack().fillna(0) * 100
        fig, ax = plt.subplots()
        state_exit.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel("% Employees")
        ax.set_title("Exit Status by State")
        st.pyplot(fig)

    with tabs[7]:
        st.subheader("Salary Distribution by Department")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='department', y='salary', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tabs[8]:
        st.subheader("Gender-wise Exit Ratio")
        gender_exit = df.groupby('gender')['exit_status'].value_counts(normalize=True).unstack().fillna(0) * 100
        st.dataframe(gender_exit.style.format("{:.2f}"))
        fig, ax = plt.subplots()
        gender_exit.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel("% Employees")
        st.pyplot(fig)

    with tabs[9]:
        st.subheader("📋 AI Executive Summary")
        numeric_summary = df.describe(include='all').to_string()
        prompt = f"""
        Given the following statistical summary of HR data, write an executive summary in 5 bullet points and 2 action recommendations.

        {numeric_summary}
        """
        if st.button("Generate Executive Summary"):
            with st.spinner("Generating insights..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"OpenAI API Error: {e}")

    with tabs[10]:
        st.subheader("📊 AI Explanation of Promotion Bias")
        #sample_stats = df[['gender', 'salary', 'performance_score', 'tenure_years', 'job_level']].groupby('gender').mean().to_string()
        sample_stats = df[['gender', 'salary', 'performance_score', 'tenure_years']].groupby('gender').mean(numeric_only=True).to_string()

        prompt = f"""
        Based on the following average HR data per gender, explain whether there might be a promotion bias and what patterns are visible:

        {sample_stats}
        """
        if st.button("Explain Promotion Bias with AI"):
            with st.spinner("Analyzing with GPT-4..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"Error: {e}")

    with tabs[11]:
        st.subheader("🔮 Simulate Scenarios with AI")
        scenario = st.text_area("Describe a hypothetical HR policy change or situation:",
                                "What if we ensure equal average salaries for men and women?")
        if st.button("Simulate with AI"):
            with st.spinner("Generating scenario analysis..."):
                try:
                    prompt = f"""
                    Given the HR data context, simulate what might happen if:
                    {scenario}

                    Provide a realistic AI-driven narrative on outcomes, risks, and benefits.
                    """
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")

    with tabs[12]:
        st.subheader("💬 Ask Anything About HR Data")
        user_question = st.text_input("Ask a question about your HR dataset:", "What is the most common reason for attrition?")
        if st.button("Answer with AI"):
            try:
                head = df.head(10).to_string()
                prompt = f"""
                Using the following sample of HR data, answer this question:

                Data:
                {head}

                Question:
                {user_question}
                """
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown(response['choices'][0]['message']['content'])
            except Exception as e:
                st.error(f"Error: {e}")
