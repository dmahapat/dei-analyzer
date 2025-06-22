# frontend/state_cluster_tab.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_state_data():
    return pd.read_csv("../data/cleaned_hr_data.csv")

def show_state_clusters():
    st.markdown("## 📍 Smart Diversity Insights per State / Region")
    st.markdown("Using AI + Clustering to detect diversity patterns across regions.")

    df = load_state_data()

    
    
    # Ensure job_level is numeric (L1, L2 → 1, 2)
    df['job_level'] = df['job_level'].str.extract('(\d+)').astype(float)

    # Convert salary and age to numeric
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows with missing crucial fields
    df = df.dropna(subset=['state_of_origin', 'gender', 'job_level', 'salary', 'age'])

    st.write("🔍 Cleaned DF Shape", df.shape)


   
    # Preprocess: Aggregate diversity features per state
    # Aggregate diversity metrics per state
    state_summary = df.groupby('state_of_origin').agg({
        'gender': lambda x: (x == 'Female').sum() / len(x),
        'job_level': 'mean',
        'salary': 'mean',
        'age': 'mean'
    }).rename(columns={
        'gender': 'female_ratio',
        'job_level': 'avg_job_level',
        'salary': 'avg_salary',
        'age': 'avg_age'
    }).reset_index()

    st.write("⚠️ Grouped Features Preview", state_summary)


    st.dataframe(state_summary)

    # Normalize features for clustering
    scaler = StandardScaler()
    features = state_summary[['female_ratio', 'avg_job_level', 'avg_salary', 'avg_age']]
    X_scaled = scaler.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    state_summary['Cluster'] = kmeans.fit_predict(X_scaled)

    # Display cluster info
    st.subheader("🧭 Clustered States (Grouped by Similar Diversity)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x=state_summary['female_ratio'],
        y=state_summary['avg_job_level'],
        hue=state_summary['Cluster'],
        palette='Set2',
        s=100
    )
    plt.xlabel("Female Ratio")
    plt.ylabel("Average Job Level")
    st.pyplot(fig)

    # After showing the scatter plot (e.g., st.pyplot(fig)), add this:
    st.markdown("### 🧠 Cluster Insight Summary")

    cluster_insights = pd.DataFrame({
        "Cluster": [0, 1, 2],
        "Insight": [
            "🟢 High hiring of women (female ratio = 1.0) but poor upward mobility (low avg job level). Investigate career advancement blockers.",
            "🟠 Male-dominated regions with higher average job levels but low female presence. Review recruitment practices.",
            "🔵 States with balanced female ratios and good job levels. Represents inclusive practices worth replicating."
        ]
    })

    # Display in a table
    st.dataframe(cluster_insights, use_container_width=True)

    # NLP-Based Summary for each cluster
    st.markdown("### 🧠 AI Insights: Suggested Regional Actions")
    for cluster in sorted(state_summary['Cluster'].unique()):
        states = state_summary[state_summary['Cluster'] == cluster]['state_of_origin'].tolist()
        high_female = state_summary[state_summary['Cluster'] == cluster]['female_ratio'].mean()
        job_level = state_summary[state_summary['Cluster'] == cluster]['avg_job_level'].mean()
        suggestion = ""

        if high_female > 0.5:
            suggestion += "- These states show strong female representation. Strengthen leadership roles.\n"
        else:
            suggestion += "- Gender balance is low. Reevaluate hiring strategies.\n"

        if job_level < 3:
            suggestion += "- Consider job elevation/retention programs.\n"
        else:
            suggestion += "- Job levels are balanced. Focus on upskilling.\n"

        st.markdown(f"**Cluster {cluster}**: `{', '.join(states)}`")
        st.markdown(f"📌 {suggestion}")
        st.markdown("---")
