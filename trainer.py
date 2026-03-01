import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import clean_text
import os

# Ensure directories exist
os.makedirs("model", exist_ok=True)

def train_mood_matrix():
    print("🚀 Starting MoodMatrix Training Pipeline...")

    # 1. Load Data
    data_path = "data/dataset.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}. Please check the path.")
        return

    df = pd.read_csv(data_path)
    
    # Handle possible empty rows or headers issues from CSV
    df = df.dropna(subset=['statement', 'status'])
    
    print(f"📊 Dataset Loaded: {df.shape[0]} rows.")
    print(f"🏷️ Classes: {df['status'].unique()}")

    # 2. Preprocessing
    print("🧹 Preprocessing text...")
    df['clean_statement'] = df['statement'].apply(lambda x: clean_text(str(x)))

    X = df['clean_statement']
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Define Pipeline and Hyperparameters
    # We will test Logistic Regression and Linear SVC
    
    pipelines = {
        'LogisticRegression': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000, multi_class='ovr'))
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC(dual=False))
        ])
    }

    params = {
        'LogisticRegression': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [5000, 10000],
            'clf__C': [0.1, 1, 10]
        },
        'LinearSVC': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [5000, 10000],
            'clf__C': [0.1, 1, 10]
        }
    }

    best_models = {}
    
    for name, pipe in pipelines.items():
        print(f"🔎 Tuning {name}...")
        grid = GridSearchCV(pipe, params[name], cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"✅ Best Params for {name}: {grid.best_params_}")
        print(f"📈 Best CV Score: {grid.best_score_:.4f}")

    # 4. Model Comparison & Evaluation
    print("\n🏁 Evaluating Models on Test Set...")
    results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    # Pick the best overall model (Logistic Regression preferred if accuracy is similar for interpretability)
    # But here we just pick the one with highest accuracy
    best_overall_name = max(results, key=results.get)
    best_overall_model = best_models[best_overall_name]
    
    print(f"\n🏆 Best Model: {best_overall_name} with Accuracy: {results[best_overall_name]:.4f}")

    # 5. Save the best pipeline
    model_save_path = "model/moodmatrix_model.joblib"
    joblib.dump(best_overall_model, model_save_path)
    print(f"💾 Best pipeline saved to {model_save_path}")

    # 6. Interpretability (for Logistic Regression specifically)
    if 'LogisticRegression' in best_models:
        lr_model = best_models['LogisticRegression']
        tfidf = lr_model.named_steps['tfidf']
        clf = lr_model.named_steps['clf']
        feature_names = tfidf.get_feature_names_out()
        
        interpret_data = {}
        for i, class_label in enumerate(clf.classes_):
            top10_idx = np.argsort(clf.coef_[i])[-10:]
            top10_features = [feature_names[idx] for idx in top10_idx]
            interpret_data[class_label] = top10_features
            
        joblib.dump(interpret_data, "model/interpretability.joblib")
        print("🧠 Interpretability data saved.")

if __name__ == "__main__":
    train_mood_matrix()
