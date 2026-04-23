
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv("data/train.csv")

print(df.head())

# =========================
# FIX YOUR DATASET STRUCTURE
# =========================

# Column 1 = label (True/False)
# Column 2 = news text

df = df.iloc[:, :2]  # take only first 2 columns

df.columns = ["label", "text"]

# Convert True/False → 1/0
df["label"] = df["label"].astype(int)

# Features & target
X = df["text"]
y = df["label"]

# Text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")