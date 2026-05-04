# Fake News Detection System

A robust Machine Learning system designed to identify and classify news articles as **Real** or **Fake**. This project leverages Natural Language Processing (NLP) techniques and various classification algorithms to provide high-accuracy predictions.

## Overview

In the era of information overload, the spread of misinformation is a significant challenge. This system provides a tool to verify the authenticity of news articles by analyzing their content. It processes raw text, extracts meaningful features using TF-IDF, and utilizes the best-performing machine learning model for classification.

## Key Features

- **Advanced Text Preprocessing**: 
  - Removal of special characters and numbers.
  - Tokenization and stopword removal (NLTK).
  - Stemming using the Porter Stemmer algorithm.
- **Feature Extraction**: Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors.
- **Model Benchmarking**: Automatically trains and compares multiple models:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Passive Aggressive Classifier
  - Decision Tree
  - Random Forest
- **Automated Selection**: Automatically saves the model with the highest accuracy for future predictions.
- **Data Visualization**: Generates detailed performance reports including:
  - Confusion Matrices for each model.
  - Model Comparison graphs (Accuracy, Precision, Recall, F1-Score).

## Technology Stack

- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **NLP Library**: NLTK (Natural Language Toolkit)
- **Serialization**: Joblib
- **Visualization**: Matplotlib, Seaborn

## Project Structure

```text
Fake News Detection/
├── data/               # Datasets (sample_news.csv)
├── models/             # Saved model and vectorizer (.pkl)
├── reports/            # Performance graphs and confusion matrices
├── src/
│   ├── preprocess.py   # Text cleaning logic
│   ├── train.py        # Model training and evaluation
│   └── predict.py      # Prediction pipeline
├── main.py             # CLI Entry point
└── requirements.txt    # Project dependencies
```

## Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

23. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The system operates in two modes: `train` and `predict`.

### 1. Training and Evaluation
To train the models on your dataset and save the best one:
```bash
python main.py --mode train
```
*This will also generate performance reports in the `reports/` folder.*

### 2. Prediction
To predict whether a specific news article is real or fake:
```bash
python main.py --mode predict --text "Paste your news article content here"
```

## Results
The system evaluates models based on Accuracy, Precision, Recall, and F1-Score. Detailed confusion matrices and comparison charts are saved in the `reports/` directory to provide insights into model performance.

---
