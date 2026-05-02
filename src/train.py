import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.preprocess import clean_text
except ImportError:
    from preprocess import clean_text

def plot_model_comparison(metrics_df, save_path):
    plt.figure(figsize=(12, 8))
    metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.ylim(0, 1.1)
    plt.title('Model Comparison - Accuracy, Precision, Recall, F1-Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Model comparison chart saved to {save_path}")

def plot_cm(y_test, y_pred, model_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(data_path, model_save_path, vectorizer_save_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Fill missing values
    df = df.fillna('')
    
    # Combine title and text for training
    print("Preprocessing text (this might take a while)...")
    df['content'] = df['title'] + " " + df['text']
    df['content'] = df['content'].apply(clean_text)
    
    X = df['content'].values
    y = df['label'].values
    
    # Feature Extraction
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    results = []
    best_model = None
    best_accuracy = 0
    
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
        print(f"Accuracy: {acc:.4f}")
        
        # Save confusion matrix
        cm_path = os.path.join(reports_dir, f'cm_{name.replace(" ", "_").lower()}.png')
        plot_cm(y_test, y_pred, name, cm_path)
        
        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # Create Comparison Report
    metrics_df = pd.DataFrame(results)
    print("\n--- Accuracy Report ---")
    print(metrics_df.to_string(index=False))
    
    # Save Model Comparison Graph
    comparison_graph_path = os.path.join(reports_dir, 'model_comparison.png')
    plot_model_comparison(metrics_df, comparison_graph_path)
    
    # Save the Best Model and Vectorizer
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    print(f"Saving best model to {model_save_path}...")
    joblib.dump(best_model, model_save_path)
    print(f"Saving vectorizer to {vectorizer_save_path}...")
    joblib.dump(vectorizer, vectorizer_save_path)
    
    print("\nTraining and evaluation completed successfully.")

if __name__ == "__main__":
    DATA_PATH = os.path.join('data', 'sample_news.csv')
    MODEL_PATH = os.path.join('models', 'fake_news_model.pkl')
    VEC_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    train_model(DATA_PATH, MODEL_PATH, VEC_PATH)
