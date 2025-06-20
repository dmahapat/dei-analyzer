from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_feedback(feedback_text):
    if not feedback_text.strip() or feedback_text.strip().lower() == "no feedback provided.":
        return ("Neutral", "No feedback to analyze.", 50)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert HR analyst. Classify sentiment, "
                        "summarize the feedback, and rate inclusiveness from 0 to 100."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analyze this employee feedback:\n\n\"{feedback_text}\""
                }
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Very basic parsing logic
        lines = content.strip().split('\n')
        sentiment = "Neutral"
        summary = ""
        score = 50

        for line in lines:
            if "sentiment" in line.lower():
                sentiment = line.split(":")[-1].strip()
            elif "summary" in line.lower():
                summary = line.split(":")[-1].strip()
            elif "score" in line.lower():
                score = int(''.join(filter(str.isdigit, line)))

        return (sentiment, summary, score)

    except Exception as e:
        print(f"❌ Error processing feedback: {e}")
        return ("Error", "Could not analyze.", 0)
