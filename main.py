import argparse
import os
from src.train import train_model
from src.predict import predict_news

def main():
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help="Mode: train or predict")
    parser.add_argument('--text', type=str, help="Text to predict (required if mode is predict)")
    
    args = parser.parse_args()
    
    DATA_PATH = os.path.join('data', 'sample_news.csv')
    MODEL_PATH = os.path.join('models', 'fake_news_model.pkl')
    VEC_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    if args.mode == 'train':
        train_model(DATA_PATH, MODEL_PATH, VEC_PATH)
    elif args.mode == 'predict':
        if not args.text:
            print("Error: Please provide --text for prediction.")
            return
        result = predict_news(args.text, MODEL_PATH, VEC_PATH)
        print(f"\nResult: {result}")

if __name__ == "__main__":
    main()
