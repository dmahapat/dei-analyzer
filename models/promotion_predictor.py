import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

def train_rf_model():
    df = pd.read_csv("../data/promotion_data.csv")

    df = df.dropna(subset=["performance_score", "tenure_years", "promotion_eligible"])
    df["promotion_eligible"] = df["promotion_eligible"].astype(int)

    X = df[["performance_score", "tenure_years"]]
    y = df["promotion_eligible"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return model, X, shap_values, explainer


# Add this to promotion_predictor.py
def get_feature_data():
    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(base_dir, "..", "data", "promotion_data.csv"))

    df = pd.read_csv(data_path)
    df["performance_score"] = pd.to_numeric(df["performance_score"], errors="coerce")
    df["tenure_years"] = pd.to_numeric(df["tenure_years"], errors="coerce")
    df = df.dropna(subset=["performance_score", "tenure_years", "promotion_eligible"])

    X = df[["performance_score", "tenure_years"]]
    return X
