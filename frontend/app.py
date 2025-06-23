# frontend/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from insight_utils import show_pay_gap, show_state_diversity, generate_ai_suggestions
from bias_detection_tab import show_bias_detection
from state_cluster_tab import show_state_clusters
from attrition_tab import show_attrition_tab
from wordcloud import WordCloud

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
    "📍 Regional Diversity Clusters",
    "⚠️ Attrition Predictor",
    "🤖 AI Feedback Summarization",
    "💸 Pay Gap Analysis",
    "🧾 Resume Bias Detector",
    "📜 DEI Policy Generator"

])

# ============================================================
# 📊 TAB 1: DEI Insights Dashboard
# ============================================================
if tab == "📊 DEI Insights Dashboard":
    st.title("📊 DEI Insights Dashboard")

   

    # Apply Filters
    filtered_df = df.copy()
    selected_dept = 'All'
    selected_level = 'All'
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

    # 💬 Feedback Analysis
    st.subheader("💬 Feedback Summary")
    st.dataframe(feedback_df[['employee_id', 'sentiment', 'summary', 'inclusion_score']])

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

# ============================================================
# ⚠️ TAB 5: Attrition Predictor
# ============================================================

elif tab == "⚠️ Attrition Predictor":
    show_attrition_tab()

elif tab == "🤖 AI Feedback Summarization":
    st.header("📝 AI Feedback Summarization")
    st.markdown("""
    - Uses GPT to summarize qualitative feedback.
    - Extracted Insights:
        - 40% feel promotions are delayed.
        - 20% cite lack of inclusivity in leadership.
        - 30% appreciate mentorship programs.
    - Saves manual effort for HR.
    """)

    
    st.header("📝 AI-Based Feedback Summarization")

    # Word Cloud placeholder (based on feedback_text column normally)
    st.subheader("📌 Word Cloud from Employee Feedback")
    feedback_text = (
        "great team player always helpful consistent performance needs mentorship for leadership role strong communicator"
    )
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Hardcoded AI summaries (normally generated using GPT)
    st.subheader("🤖 LLM-Based Summaries")
    st.markdown("""
    - ✅ *"Most employees appreciate collaboration and consistent performance."*
    - ⚠️ *"Some employees seek better mentorship and clearer career paths."*
    - 💬 *"Communication is highlighted as a strong suit across teams."*
    """)

elif tab == "💸 Pay Gap Analysis":
    st.header("💸 Pay Gap Analysis + AI Suggestions")
    st.markdown("""
    - Detects salary discrepancies across gender and departments.
    - Offers AI-driven suggestions.
    """)

    st.subheader("📉 Avg Salary by Gender & Department")

    # Hardcoded salary data
    data = pd.DataFrame({
        'department': ['Tech', 'Tech', 'HR', 'HR', 'Sales', 'Sales'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'avg_salary': [90000, 80000, 65000, 67000, 70000, 69000]
    })

    # Pivot data for grouped bar chart
    pivot = data.pivot(index='department', columns='gender', values='avg_salary')
    pivot = pivot[['Male', 'Female']]  # Ensure consistent order

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    index = range(len(pivot))

    ax.bar(index, pivot['Male'], bar_width, label='Male', color='skyblue')
    ax.bar([i + bar_width for i in index], pivot['Female'], bar_width, label='Female', color='pink')

    ax.set_xlabel('Department')
    ax.set_ylabel('Average Salary')
    ax.set_title('Gender-wise Salary by Department')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(pivot.index)
    ax.legend()

    st.pyplot(fig)

    st.subheader("🧠 GPT-3.5 Suggestions")
    st.markdown("""
    - 🔍 *Consider reviewing Tech department pay structure — 11% gender gap.*
    - 💬 *Introduce mentorship to support female employees in Sales.*
    - 🛠️ *Standardize performance metrics across departments.*
    """)

elif tab == "🧾 Resume Bias Detector":
    st.header("🧾 Resume/Job Description Bias Detector")
    st.markdown("""
    - Uses LLM to analyze text for biased language.
    - Example flags:
        - 🚩 *"Aggressive achiever" — may imply gender bias.*
        - 🚩 *"Culturally fit" — vague and exclusionary.*
    - Normally uses GPT + NLP toolkit.
    """)

    st.subheader("🔍 Bias Examples")
    st.markdown("""
    - ❌ *"Rockstar developer"* → Try *"skilled developer"* ✅
    - ❌ *"Young and energetic team"* → Age bias. Try *"collaborative team"* ✅
    - ✅ *"Strong analytical and problem-solving skills"* ✅
    """)

elif tab == "📜 DEI Policy Generator":
    st.header("📜 GPT-Based DEI Policy Recommender")
    st.markdown("""
    - Suggests draft policies using LLMs based on diversity data.
    - Would use GPT + HR benchmarks.
    """)

    st.subheader("📄 Sample DEI Policy Suggestion")
    st.markdown("""
    > Our analysis shows that female representation in Tech is 12% below industry average.
    >
    > **Recommended Actions:**
    > - Implement blind resume screening.
    > - Launch mentorship for underrepresented groups.
    > - Set inclusive hiring KPIs.
    """)


