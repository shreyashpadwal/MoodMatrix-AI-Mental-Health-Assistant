import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import clean_text
import os
import json
from datetime import datetime
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs("model", exist_ok=True)



def train_mood_matrix():
    logger.info("🚀 Starting MoodMatrix Training Pipeline...")

    # 1. Load Data
    data_path = "data/dataset.csv"
    if not os.path.exists(data_path):
        logger.error(f"❌ Error: Dataset not found at {data_path}. Please check the path.")
        return

    df = pd.read_csv(data_path)
    
    logger.info("📊 Original Class Distribution:")
    logger.info("\n" + str(df['status'].value_counts()))
    
    # Handle empty rows and deduplicate
    df = df.dropna(subset=['statement', 'status']).drop_duplicates(subset=['statement'])
    dataset_size = df.shape[0]
    
    logger.info(f"\n📊 Dataset Loaded (after dedup): {dataset_size} rows.")
    logger.info("📊 Post-Dedup Class Distribution:")
    logger.info("\n" + str(df['status'].value_counts()))
    
    classes = list(df['status'].unique())
    logger.info(f"🏷️ Classes: {classes}")

    # 2. Preprocessing
    logger.info("🧹 Preprocessing text...")
    df['clean_statement'] = df['statement'].apply(lambda x: clean_text(str(x)))

    X = df['clean_statement']
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Define Pipeline and Hyperparameters
    pipelines = {
        'LogisticRegression': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC(dual=False, random_state=42))
        ]),
        'SGDClassifier': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SGDClassifier(loss='log_loss', max_iter=1000, random_state=42))
        ])
    }

    params = {
        'LogisticRegression': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [10000],
            'tfidf__min_df': [1],
            'tfidf__sublinear_tf': [True],
            'clf__C': [1],
            'clf__class_weight': ['balanced']
        },
        'LinearSVC': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [10000],
            'tfidf__min_df': [2],
            'tfidf__sublinear_tf': [True],
            'clf__C': [0.1],
            'clf__class_weight': ['balanced']
        },
        'SGDClassifier': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [10000],
            'tfidf__min_df': [1],
            'tfidf__sublinear_tf': [True],
            'clf__alpha': [0.0001],
            'clf__class_weight': ['balanced']
        }
    }

    best_models = {}
    
    for name, pipe in pipelines.items():
        logger.info(f"🔎 Tuning {name}...")
        grid = GridSearchCV(pipe, params[name], cv=3, n_jobs=-1, verbose=1, scoring='f1_macro')
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        logger.info(f"✅ Best Params for {name}: {grid.best_params_}")
        logger.info(f"📈 Best CV Score (Macro F1): {grid.best_score_:.4f}")

    # 4. Evaluate TF-IDF Models
    logger.info("\n🏁 Evaluating TF-IDF Models on Test Set...")
    results = {}
    reports = {}
    f1_scores = {}
    
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        reports[name] = classification_report(y_test, y_pred, output_dict=True)
        f1_scores[name] = reports[name]['macro avg']['f1-score']
        logger.info(f"\n--- {name} ---")
        logger.info(f"Accuracy: {acc:.4f} | Macro F1: {f1_scores[name]:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred, digits=3))

    best_tfidf_name = max(f1_scores, key=f1_scores.get)
    if 'LogisticRegression' in f1_scores and best_tfidf_name != 'LogisticRegression':
        if f1_scores['LogisticRegression'] >= f1_scores[best_tfidf_name] - 0.01:
            best_tfidf_name = 'LogisticRegression'
            logger.info("Preferred LogisticRegression for predict_proba and interpretability due to similar F1 score.")

    best_tfidf_model = best_models[best_tfidf_name]
    logger.info(f"\n🏆 Best TF-IDF Model: {best_tfidf_name} with Macro F1: {f1_scores[best_tfidf_name]:.4f}")
    
    joblib.dump(best_tfidf_model, "model/moodmatrix_tfidf.joblib")
    logger.info("💾 Saved best TF-IDF model to model/moodmatrix_tfidf.joblib")

    # 5. Final Model Selection (TF-IDF wins)
    best_overall_model = best_tfidf_model
    best_overall_name = best_tfidf_name
    best_overall_acc = results[best_tfidf_name]
    best_overall_f1 = f1_scores[best_tfidf_name]
    best_overall_report = reports[best_tfidf_name]
    best_params = best_models[best_tfidf_name].named_steps['clf'].get_params() if hasattr(best_models[best_tfidf_name].named_steps['clf'], 'get_params') else "N/A"
    
    logger.info(f"\n👑 OVERALL BEST MODEL: {best_overall_name} with Macro F1: {best_overall_report['macro avg']['f1-score']:.4f}")

    model_save_path = "model/moodmatrix_model.joblib"
    joblib.dump(best_overall_model, model_save_path)
    logger.info(f"💾 Best overall pipeline saved to {model_save_path}")

    # Confusion matrix for overall best
    y_pred_best = best_overall_model.predict(X_test)
        
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {best_overall_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png')
    plt.close()

    # Normal vs rest confusion matrix for the best model
    y_test_binary = (y_test == 'Normal').astype(int)
    y_pred_binary = (np.array(y_pred_best) == 'Normal').astype(int)
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    logger.info("\n📊 Normal vs Rest Confusion Matrix:")
    logger.info(f"\n{cm_binary}")
    logger.info("Rows: Actual (0=Rest, 1=Normal), Cols: Predicted (0=Rest, 1=Normal)")

    # 6. Save Metrics JSON
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": best_overall_name,
        "overall_accuracy": best_overall_acc,
        "dataset_size": int(dataset_size),
        "hyperparameters": str(best_params),
        "per_class_metrics": {
            cls: {
                "precision": best_overall_report[cls]["precision"],
                "recall": best_overall_report[cls]["recall"],
                "f1-score": best_overall_report[cls]["f1-score"]
            } for cls in classes if cls in best_overall_report
        }
    }
    
    with open("model/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    logger.info("📊 Saved metrics to model/metrics.json")
    
    # Interpretability (only for TF-IDF LogisticRegression)
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
        logger.info("🧠 Interpretability data saved.")

if __name__ == "__main__":
    train_mood_matrix()
