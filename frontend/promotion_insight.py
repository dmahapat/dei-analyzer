


import streamlit as st
import matplotlib.pyplot as plt
import shap
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.promotion_predictor import train_rf_model, get_feature_data

# Step 1: Train the model and get feature data
model, X = train_rf_model(), get_feature_data()

# Step 2: Layout setup
st.markdown("## 💡 AI-Based Promotion Bias Detection")
st.markdown("#### 🔍 SHAP Summary (Feature Impact on Promotion Prediction)")
st.markdown("---")

# Step 3: SHAP explanation
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Step 4: Matplotlib fig fix to avoid yellow warning
fig, ax = plt.subplots(figsize=(10, 3))
shap.plots.bar(shap_values, max_display=5, show=False)
st.pyplot(fig)

# Optional styling
st.markdown("""
<style>
h1, h2, h3, h4, h5 {
    font-family: 'Segoe UI', sans-serif;
    color: #333;
}
</style>
""", unsafe_allow_html=True)
