
from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Small training dataset
data = [
    ("I love this application", "positive"),
    ("This is an amazing project", "positive"),
    ("The service works perfectly", "positive"),
    ("I am very happy", "positive"),
    ("Excellent performance", "positive"),
    ("This product is fantastic", "positive"),
    ("The experience was wonderful", "positive"),
    ("Everything is working great", "positive"),
    ("I really enjoyed this", "positive"),
    ("The results are impressive", "positive"),
    ("This is a good solution", "positive"),
    ("The application is fast and reliable", "positive"),
    ("I hate this application", "negative"),
    ("This project is terrible", "negative"),
    ("The service is broken", "negative"),
    ("I am very disappointed", "negative"),
    ("Poor performance", "negative"),
    ("This product is awful", "negative"),
    ("The experience was horrible", "negative"),
    ("Nothing works correctly", "negative"),
    ("I really disliked this", "negative"),
    ("The results are disappointing", "negative"),
    ("This is a bad solution", "negative"),
    ("The application is slow and unreliable", "negative"),
]

texts, labels = zip(*data)

# Create and train the ML pipeline
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)

model.fit(texts, labels)

# Save the trained model
model_path = Path(__file__).resolve().parent / "app" / "model.joblib"
joblib.dump(model, model_path)

print("Model trained successfully!")
print(f"Saved to: {model_path}")
