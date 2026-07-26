# 🧠 MoodMatrix: AI Mental Health Assistant

MoodMatrix is a technically strong, production-style NLP system designed for mental health text classification. It uses advanced machine learning pipelines to categorize user thoughts into multiple mental health-related categories.

## 🚀 Key Features
- **Scikit-learn Pipeline & Embeddings**: Integrated TF-IDF Vectorization or Sentence Transformers with models like Logistic Regression, Linear SVM, and SGDClassifier.
- **Hyperparameter Tuning**: Optimized using `GridSearchCV` for best performance.
- **Class Imbalance Handling**: Addresses minority classes via balanced weights to improve recall.
- **Data Hygiene**: Pre-split deduplication ensures no data leakage between train and test sets.
- **Modern UI**: Clean, multi-page Streamlit interface with sidebar navigation reading dynamic metrics.
- **Modular Codebase**: Separated concerns across `trainer.py`, `preprocessing.py`, and `mood_app.py`.

## 🏗️ Architecture
```mermaid
graph TD
    A[User Input] --> B[NLP Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[ML Classifier]
    D --> E[Prediction + Confidence]
    D --> F[Model Interpretability]
```

- **Preprocessing**: Lowercasing, robust punctuation preservation (`!`, `?`), stopword removal, and NLTK lemmatization.
- **Model**: Scikit-Learn Pipeline (`TfidfVectorizer` + `LogisticRegression`/`LinearSVC`) or `SentenceTransformers` embeddings.
- **Optimization**: `GridSearchCV` tuning hyperparameters with rigorous cross-validation.

## 📁 Project Structure
```
MoodMatrix/
│
├── mood_app.py         # Main Streamlit application
├── trainer.py          # Automated training & tuning pipeline
├── preprocessing.py    # Reusable NLP preprocessing functions
├── data/
│   └── dataset.csv     # Training dataset
├── model/
│   ├── moodmatrix_model.joblib  # Trained pipeline
│   └── interpretability.joblib   # Extracted feature importance
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🛠️ How to Run Locally

1. **Clone the repository** (if applicable) and navigate to the directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model**:
   ```bash
   python trainer.py
   ```
4. **Launch the app**:
   ```bash
   streamlit run mood_app.py
   ```

## 📊 Model Performance
The pipeline tracks metrics automatically in `model/metrics.json` and evaluates the best overall model (TF-IDF vs Embeddings).
Key improvements in the latest version:
1. **Deduplication Fix**: Removed 1,969 duplicate rows prior to splitting, fixing a data leakage issue that artificially inflated test accuracy.
2. **Class Imbalance**: Added `class_weight='balanced'`, significantly boosting recall for minority classes like Personality disorder and Stress.
3. **Advanced Models**: Integrated `sentence-transformers` for dense semantic embeddings.

## 📈 Future Enhancements
- **REST API**: Convert the system into a FastAPI service for integration with mobile/web apps.
- **Dockerization**: Create a Docker container for seamless deployment.
- **Cloud Deployment**: Deploy on Streamlit Cloud or AWS/GCP.

---
*Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.*
