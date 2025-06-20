import pandas as pd
import numpy as np
import os


# Define path to CSV
file_path = "../data/sample_hr_data.csv"

# Load the data
try:
    df = pd.read_csv(file_path)
    print("✅ Raw data loaded successfully.\n")
except Exception as e:
    print("❌ Error loading data:", e)
    exit()


print("🔍 Data preview:\n", df.head())
print("\n🧾 Column Summary:\n", df.info())
print("\n📊 Null counts:\n", df.isnull().sum())


gender_map = {
    "M": "Male",
    "F": "Female",
    "m": "Male",
    "f": "Female",
    "male": "Male",
    "female": "Female",
    "nb": "Non-binary",
    "Nonbinary": "Non-binary"
}
df['gender'] = df['gender'].str.strip().str.capitalize().map(gender_map).fillna(df['gender'])

df['state_of_origin'] = df['state_of_origin'].str.strip().str.title()

valid_ratings = ['Exceeds Expectations', 'Meets Expectations', 'Below Expectations']
df['performance_score'] = df['performance_score'].apply(lambda x: x if x in valid_ratings else 'Meets Expectations')

numerical_cols = ['age', 'salary', 'tenure_years']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Invalid strings → NaN
    df[col] = df[col].fillna(df[col].median())

df['feedback_text'] = df['feedback_text'].fillna("No feedback provided.")


before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"\n🧹 Dropped {before - after} duplicate records.")

output_path = "../data/cleaned_hr_data.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved to {output_path}")


