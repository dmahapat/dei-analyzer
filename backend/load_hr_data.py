import pandas as pd

# Load data
file_path = "../data/sample_hr_data.csv"
try:
    df = pd.read_csv(file_path)
    print("\n✅ Sample HR data loaded successfully.\n")
    print(df.head())  # Print the first 5 rows
except Exception as e:
    print("\n❌ Failed to load the HR data.")
    print("Error:", e)
