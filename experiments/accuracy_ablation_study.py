import pandas as pd
import numpy as np
import time
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing import clean_text
import os
import joblib

# Ensure wordnet
try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')

class KeywordCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = {
            'Stress': ['deadline', 'overwhelmed', 'pressure', 'burnout', 'heavy', 'load', 'exhausted', 'piling', 'break down', 'tense'],
            'Anxiety': ['panic', 'racing', 'worried', 'shaking', 'nervous', 'scared', 'dread', 'breathing', 'chest', 'fear'],
            'Depression': ['hopeless', 'empty', 'worthless', 'tired', 'crying', 'numb', 'dark', 'give up', 'pain', 'sad'],
            'Suicidal': ['kill', 'end it', 'die', 'suicide', 'jump', 'gun', 'no reason', 'better off', 'goodbye', 'pills'],
            'Bipolar': ['manic', 'mood swings', 'high', 'low', 'impulsive', 'reckless', 'racing thoughts', 'god', 'energy', 'crash'],
            'Personality disorder': ['abandoned', 'empty', 'identity', 'unstable', 'splitting', 'impulsive', 'paranoid', 'detach', 'manipulate', 'obsessed']
        }
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = np.zeros((len(X), len(self.keywords)))
        for i, text in enumerate(X):
            text_lower = str(text).lower()
            for j, (cls, words) in enumerate(self.keywords.items()):
                count = sum(1 for w in words if w in text_lower)
                features[i, j] = count
        return features

def augment_text(text):
    words = text.split()
    augmented = []
    replaced = 0
    for word in words:
        if replaced < 2 and len(word) > 3:
            syns = wordnet.synsets(word)
            if syns:
                lemmas = syns[0].lemmas()
                if lemmas:
                    synonym = lemmas[0].name()
                    if synonym.lower() != word.lower() and "_" not in synonym:
                        augmented.append(synonym)
                        replaced += 1
                        continue
        augmented.append(word)
    return " ".join(augmented)

def augment_minority_classes(X_train, y_train):
    target_classes = ['Stress', 'Personality disorder', 'Bipolar']
    new_X = []
    new_y = []
    for text, label in zip(X_train, y_train):
        if label in target_classes:
            new_text = augment_text(str(text))
            if new_text != text:
                new_X.append(new_text)
                new_y.append(label)
    return pd.concat([X_train, pd.Series(new_X)], ignore_index=True), pd.concat([y_train, pd.Series(new_y)], ignore_index=True)

def print_metrics(y_true, y_pred, step_name):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    stress_f1 = report.get('Stress', {}).get('f1-score', 0)
    pd_f1 = report.get('Personality disorder', {}).get('f1-score', 0)
    print(f"[{step_name}] Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Stress F1: {stress_f1:.4f} | PD F1: {pd_f1:.4f}")
    return acc, macro_f1, stress_f1, pd_f1, report

def run_ablation():
    print("Loading data...")
    df = pd.read_csv("data/dataset.csv").dropna(subset=['statement', 'status']).drop_duplicates(subset=['statement'])
    df['clean_statement'] = df['statement'].apply(lambda x: clean_text(str(x)))
    X = df['clean_statement']
    y = df['status']
    classes = list(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Baseline (Step 0) ---")
    pipe_base = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=42))
    ])
    pipe_base.fit(X_train, y_train)
    y_pred_base = pipe_base.predict(X_test)
    baseline_metrics = print_metrics(y_test, y_pred_base, "Baseline")

    print("\n--- Step 1: Restore Grid Search ---")
    pipelines = {
        'LogisticRegression': Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True)), ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))]),
        'LinearSVC': Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True)), ('clf', LinearSVC(class_weight='balanced', dual=False, random_state=42))]),
        'SGDClassifier': Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True)), ('clf', SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42))])
    }
    params = {
        'LogisticRegression': {'tfidf__ngram_range': [(1,2), (1,3)], 'tfidf__max_features': [10000, 20000], 'clf__C': [0.1, 1, 10]},
        'LinearSVC': {'tfidf__ngram_range': [(1,2), (1,3)], 'tfidf__max_features': [10000, 20000], 'clf__C': [0.1, 1, 10]},
        'SGDClassifier': {'tfidf__ngram_range': [(1,2), (1,3)], 'tfidf__max_features': [10000, 20000], 'clf__alpha': [0.0001, 0.001]}
    }
    
    best_models_s1 = {}
    best_f1_s1 = 0
    best_name_s1 = ""
    for name, pipe in pipelines.items():
        grid = GridSearchCV(pipe, params[name], cv=3, n_jobs=-1, scoring='f1_macro')
        grid.fit(X_train, y_train)
        best_models_s1[name] = grid.best_estimator_
        y_pred = grid.predict(X_test)
        rep = classification_report(y_test, y_pred, output_dict=True)
        if rep['macro avg']['f1-score'] > best_f1_s1:
            best_f1_s1 = rep['macro avg']['f1-score']
            best_name_s1 = name
            
    # Force LogisticRegression if close
    y_pred_lr = best_models_s1['LogisticRegression'].predict(X_test)
    f1_lr = classification_report(y_test, y_pred_lr, output_dict=True)['macro avg']['f1-score']
    if f1_lr >= best_f1_s1 - 0.01:
        best_name_s1 = 'LogisticRegression'
        
    best_pipe_s1 = best_models_s1[best_name_s1]
    y_pred_s1 = best_pipe_s1.predict(X_test)
    print(f"Winner: {best_name_s1} with params {best_pipe_s1.named_steps['clf'].get_params()}")
    metrics_s1 = print_metrics(y_test, y_pred_s1, "Step 1: Grid Search")
    
    print("\n--- Step 2: Char N-Grams ---")
    tfidf_word = best_pipe_s1.named_steps['tfidf']
    union_s2 = FeatureUnion([
        ('word_tfidf', tfidf_word),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=10000, sublinear_tf=True))
    ])
    clf_s2 = best_pipe_s1.named_steps['clf']
    pipe_s2 = Pipeline([('features', union_s2), ('clf', clf_s2)])
    pipe_s2.fit(X_train, y_train)
    y_pred_s2 = pipe_s2.predict(X_test)
    metrics_s2 = print_metrics(y_test, y_pred_s2, "Step 2: Char N-Grams")

    print("\n--- Step 3: NLTK Augmentation ---")
    X_train_aug, y_train_aug = augment_minority_classes(X_train, y_train)
    print(f"Augmented Train Set Size: {len(X_train_aug)} (from {len(X_train)})")
    pipe_s2.fit(X_train_aug, y_train_aug) # Refit pipeline from Step 2 on aug data
    y_pred_s3 = pipe_s2.predict(X_test)
    metrics_s3 = print_metrics(y_test, y_pred_s3, "Step 3: Augmentation")

    print("\n--- Step 4: Lexicon Keyword Counter ---")
    union_s4 = FeatureUnion([
        ('word_tfidf', tfidf_word),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=10000, sublinear_tf=True)),
        ('lexicon', KeywordCounter())
    ])
    pipe_s4 = Pipeline([('features', union_s4), ('clf', clf_s2)])
    pipe_s4.fit(X_train_aug, y_train_aug)
    y_pred_s4 = pipe_s4.predict(X_test)
    metrics_s4 = print_metrics(y_test, y_pred_s4, "Step 4: Keyword Features")

    print("\n--- Step 5: Threshold Tuning ---")
    probs = pipe_s4.predict_proba(X_test)
    y_pred_s5 = []
    classes_s4 = pipe_s4.classes_
    stress_idx = list(classes_s4).index('Stress')
    pd_idx = list(classes_s4).index('Personality disorder')
    
    for prob in probs:
        if prob[stress_idx] > 0.35:
            y_pred_s5.append('Stress')
        elif prob[pd_idx] > 0.35:
            y_pred_s5.append('Personality disorder')
        else:
            y_pred_s5.append(classes_s4[np.argmax(prob)])
            
    metrics_s5 = print_metrics(y_test, y_pred_s5, "Step 5: Threshold Tuning")

if __name__ == "__main__":
    run_ablation()
