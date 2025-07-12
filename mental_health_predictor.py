# mental_health_predictor.py

from transformers import pipeline

# Load pre-trained sentiment-analysis pipeline from Hugging Face
classifier = pipeline("sentiment-analysis")

print("🧠 AI-Based Mental Health Text Sentiment Checker")
print("Type something emotional and press Enter:\n")

while True:
    text = input("You: ")
    if text.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)

    print(f"Prediction: {label} ({score}%)\n")
