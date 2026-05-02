import joblib
import os
try:
    from src.preprocess import clean_text
except ImportError:
    from preprocess import clean_text

def predict_news(text, model_path, vectorizer_path):
    # Load model and vectorizer
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return "Error: Model or Vectorizer files not found. Please run train.py first."
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Preprocess the input text
    cleaned_text = clean_text(text)
    
    # Transform using vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(vectorized_text)
    
    return "Real News" if prediction[0] == 1 else "Fake News"

if __name__ == "__main__":
    MODEL_PATH = os.path.join('models', 'fake_news_model.pkl')
    VEC_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    test_text = "Scientists have discovered a new planet orbiting the nearest star."
    result = predict_news(test_text, MODEL_PATH, VEC_PATH)
    print(f"News: {test_text}")
    print(f"Prediction: {result}")
    
    fake_text = "Magic potion discovered in Amazon rainforest that grants invisibility."
    result_fake = predict_news(fake_text, MODEL_PATH, VEC_PATH)
    print(f"\nNews: {fake_text}")
    print(f"Prediction: {result_fake}")
