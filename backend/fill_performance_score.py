import numpy as np
import pandas as pd

df = pd.read_csv("../data/cleaned_hr_data.csv")

# Randomly assign performance score between 1 and 5
np.random.seed(42)
df["performance_score"] = np.random.randint(1, 6, size=len(df))

df.to_csv("../data/cleaned_hr_data.csv", index=False)
