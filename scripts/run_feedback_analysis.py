# scripts/run_feedback_analysis.py

import pandas as pd
import time
from models.feedback_summarizer import analyze_feedback

df = pd.read_csv("data/cleaned_hr_data.csv")
feedback_results = []

for i, row in df.iterrows():
    print(f"🔍 Analyzing feedback for employee {row['employee_id']}...")
    sentiment, summary, score = analyze_feedback(row['feedback_text'])
    feedback_results.append({
        "employee_id": row['employee_id'],
        "sentiment": sentiment,
        "summary": summary,
        "inclusion_score": score
    })
    time.sleep(1)  # To avoid API rate limits

result_df = pd.DataFrame(feedback_results)
result_df.to_csv("reports/feedback_analysis.csv", index=False)
print("✅ Feedback analysis saved to reports/feedback_analysis.csv")
