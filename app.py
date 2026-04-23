import pickle

# Load model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    return "FAKE NEWS 🔴" if prediction[0] == 0 else "REAL NEWS 🟢"

# Take input
news = input("Enter news text: ")

# Predict
print("\nResult:", predict_news(news))