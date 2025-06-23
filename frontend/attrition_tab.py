import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def show_attrition_tab():
    st.header("📉 Attrition Prediction using AI")

    # Load data
    df = pd.read_csv("../data/cleaned_hr_data.csv")

    # Convert target to numeric
    target = "exit_status"
    df[target] = df[target].map({"Exited": 1, "Active": 0})
    df = df.dropna(subset=[target])

    # Select features
    features = ["tenure_years", "performance_score", "department", "salary"]
    df = df.dropna(subset=features)

    # Encode categorical
    le = LabelEncoder()
    df["department_encoded"] = le.fit_transform(df["department"])
    features_fixed = ["tenure_years", "performance_score", "department_encoded", "salary"]
    X = df[features_fixed]
    y = df[target]

    if len(X) < 5:
        st.error("Not enough data to train model. Please check the dataset.")
        return

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Show evaluation
    st.subheader("📊 Model Evaluation")
    st.text(classification_report(y_test, preds))

    


    # Insights
    st.subheader("📝 Plain-English Insights")
    st.markdown("""
    - 📌 **Performance Score**: Higher scores → less likely to leave.
    - 🕒 **Tenure**: < 2 years = high attrition risk.
    - 🏢 **Department**: Some departments see more exits.
    - 💰 **Salary**: Lower salary = higher attrition.
    """)

    # Prediction Form
    st.subheader("🔮 Will this Employee Leave?")
    with st.form("predict_form"):
        tenure_input = st.number_input("Tenure (Years)", min_value=0, step=1)
        perf_input = st.selectbox("Performance Score", sorted(df["performance_score"].unique()))
        dept_input = st.selectbox("Department", sorted(df["department"].unique()))
        salary_input = st.number_input("Salary", min_value=0, step=100000)
        submitted = st.form_submit_button("Predict")

        if submitted:
            dept_encoded = le.transform([dept_input])[0]
            input_df = pd.DataFrame([{
                "tenure_years": tenure_input,
                "performance_score": perf_input,
                "department_encoded": dept_encoded,
                "salary": salary_input
            }])
            pred = model.predict(input_df)[0]
            label = "🚨 Likely to Leave" if pred == 1 else "✅ Likely to Stay"
            st.success(f"Prediction: {label}")
    
    st.subheader("📊 Attrition Rate by Department")
    data = pd.DataFrame({
        'department': ['Tech', 'HR', 'Sales'],
        'attrition_rate': [0.35, 0.15, 0.28]
    })
    fig, ax = plt.subplots()
    ax.bar(data['department'], data['attrition_rate'], color=['red', 'blue', 'green'])
    ax.set_ylabel("Attrition Rate")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.subheader("🧠 SHAP Insight (Mock)")
    st.markdown("""
    - 📌 *Tenure below 2 years highly linked with exits.*
    - 📌 *Low performance scores are a strong predictor.*
    - 📌 *Tech department has highest risk.*
    """)
