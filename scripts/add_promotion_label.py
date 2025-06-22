import pandas as pd
import numpy as np

# Step 1: Load the cleaned HR data
df = pd.read_csv("../data/cleaned_hr_data.csv")

print("Original data shape:", df.shape)
print("Preview of data:")
print(df.head())

# Step 2: Convert relevant columns to numeric
df["performance_score"] = pd.to_numeric(df["performance_score"], errors="coerce")
df["tenure_years"] = pd.to_numeric(df["tenure_years"], errors="coerce")


print(df[["performance_score", "tenure_years"]].isnull().sum())

# Step 3: Seed for reproducibility
np.random.seed(42)

# Step 4: Add 'promotion_eligible' based on logic
df["promotion_eligible"] = np.where(
    (df["performance_score"] >= 4) & (df["tenure_years"] > 2),
    np.random.choice([1, 0], size=len(df), p=[0.7, 0.3]),
    np.random.choice([1, 0], size=len(df), p=[0.3, 0.7])
)

# Step 5: Save the new dataset to promotion_data.csv
df.to_csv("../data/promotion_data.csv", index=False)
print("✅ 'promotion_eligible' column added and saved to data/promotion_data.csv")
