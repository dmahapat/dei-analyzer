import pandas as pd
import numpy as np


def calculate_dei_score(df):
    # Placeholder for final score breakdown
    scores = {}

    # -----------------------------
    # 1. DIVERSITY SCORE
    # -----------------------------
    gender_diversity = df['gender'].value_counts(normalize=True)
    gender_score = 100 - abs(gender_diversity.get('Male', 0) - gender_diversity.get('Female', 0)) * 100

    state_distribution = df['state_of_origin'].value_counts(normalize=True)
    ideal_share = 1 / len(state_distribution)
    state_score = 100 - sum(abs(state_distribution - ideal_share)) * 100

    diversity_score = round((gender_score + state_score) / 2, 2)
    scores['diversity'] = diversity_score

    # -----------------------------
    # 2. EQUITY SCORE (Pay gap)
    # -----------------------------
    avg_salary = df['salary'].mean()
    gender_salary = df.groupby('gender')['salary'].mean()
    pay_gap = abs(gender_salary.get('Male', avg_salary) - gender_salary.get('Female', avg_salary))

    pay_gap_percent = pay_gap / avg_salary
    equity_score = round(max(0, 100 - (pay_gap_percent * 100)), 2)
    scores['equity'] = equity_score

    # -----------------------------
    # 3. INCLUSION SCORE (Tenure)
    # -----------------------------
    avg_tenure = df['tenure_years'].mean()
    std_tenure = df['tenure_years'].std()

    # Assume healthy inclusion if std deviation isn't high
    inclusion_score = round(max(0, 100 - (std_tenure / avg_tenure * 100)), 2)
    scores['inclusion'] = inclusion_score

    # -----------------------------
    # 4. FINAL DEI SCORE
    # -----------------------------
    overall_score = round(np.mean(list(scores.values())), 2)
    scores['overall'] = overall_score

    return scores


if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_hr_data.csv")
    result = calculate_dei_score(df)
    print("\n✅ DEI Score Breakdown:")
    for k, v in result.items():
        print(f"{k.capitalize():<10}: {v}")

    pd.DataFrame([result]).to_csv("../reports/dei_score.csv", index=False)
    print("\n📝 Saved DEI score to '../reports/dei_score.csv'")

