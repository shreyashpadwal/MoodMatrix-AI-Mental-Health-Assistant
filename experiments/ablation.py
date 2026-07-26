import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from preprocessing import clean_text

print("Loading data...")
df = pd.read_csv("data/dataset.csv")
df = df.dropna(subset=['statement', 'status']).drop_duplicates(subset=['statement'])
X_raw = df['statement']
y = df['status']

def run_ablation(remove_stopwords):
    print(f"\nProcessing (remove_stopwords={remove_stopwords})...")
    X = X_raw.apply(lambda x: clean_text(str(x), remove_stopwords=remove_stopwords, preserve_punctuation=True))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = LinearSVC(class_weight='balanced', dual=False, random_state=42)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    
    print("Training...")
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 (remove_stopwords={remove_stopwords}): {macro_f1:.4f}")

run_ablation(True)
run_ablation(False)
