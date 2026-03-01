import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import clean_text
import os

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="MoodMatrix - AI Mental Health Assistant", layout="wide")

# -----------------------
# Load Model & Resources
# -----------------------
@st.cache_resource
def load_resources():
    model = joblib.load("model/moodmatrix_model.joblib")
    try:
        interpret = joblib.load("model/interpretability.joblib")
    except:
        interpret = None
    return model, interpret

model, interpret_data = load_resources()

# Colors for classes
class_colors = {
    "Normal": "#2ecc71",
    "Depression": "#e74c3c",
    "Anxiety": "#e67e22",
    "Stress": "#9b59b6",
    "Bipolar": "#3498db",
    "Suicidal": "#8e2c2c",
    "Personality disorder": "#f78fb3"
}

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("🧠 MoodMatrix")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔮 Prediction", "📊 Model Performance", "🧠 Interpretability", "ℹ️ About"])

# -----------------------
# Prediction Page
# -----------------------
if page == "🔮 Prediction":
    st.title("🔮 Mental Health Prediction")
    st.write("MoodMatrix uses advanced NLP to understand the emotions behind your words.")
    
    user_input = st.text_area("How are you feeling today?", height=150, placeholder="Type your thoughts here...")
    
    if st.button("Analyze Mood"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            cleaned_text = clean_text(user_input)
            prediction = model.predict([cleaned_text])[0]
            
            # Probability calculation
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([cleaned_text])[0]
                classes = model.classes_
            else:
                # Handle models without predict_proba (like LinearSVC) using decision_function if possible
                if hasattr(model, "decision_function"):
                    decision = model.decision_function([cleaned_text])[0]
                    exp_scores = np.exp(decision - np.max(decision))
                    probs = exp_scores / exp_scores.sum()
                    classes = model.classes_
                else:
                    probs = None
            
            st.markdown(f"### Result: <span style='color:{class_colors.get(prediction, '#ffffff')}'>{prediction}</span>", unsafe_allow_html=True)
            
            if probs is not None:
                # Create a bar chart for confidence
                prob_df = pd.DataFrame({"Category": classes, "Confidence": probs * 100})
                prob_df = prob_df.sort_values(by="Confidence", ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:{class_colors.get(prediction, '#34495e')}; padding:20px; border-radius:15px; color:white;">
                        <h2 style="margin:0; text-align:center;">{prediction}</h2>
                        <p style="margin:5px; text-align:center; font-size:1.2em;">Confidence: {prob_df.iloc[0]['Confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.barplot(x="Confidence", y="Category", data=prob_df, palette=[class_colors.get(c, "gray") for c in prob_df["Category"]], ax=ax)
                    ax.set_title("Probability Distribution")
                    st.pyplot(fig)

# -----------------------
# Model Performance Page
# -----------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance & Insights")
    
    st.markdown("""
    The current model is a **Scikit-learn Pipeline** consisting of:
    1. **TF-IDF Vectorizer**: Converts text to weighted numerical features.
    2. **Classifier**: Optimized via GridSearchCV for maximum accuracy.
    """)
    
    st.subheader("Model Configuration")
    st.markdown("""
    | Component | Details |
    |:---|:---|
    | **Step 1: Vectorizer** | `TfidfVectorizer` — N-grams (1,2), up to 10,000 features |
    | **Step 2: Classifier** | `LogisticRegression` — multi-class (OvR), optimized C |
    | **Tuning Method** | `GridSearchCV` — 3-fold cross-validation, 36 configurations |
    | **Classes** | Normal, Depression, Anxiety, Stress, Bipolar, Suicidal, Personality Disorder |
    | **Output** | Class label + `predict_proba()` confidence percentages |
    """)

    st.subheader("How it Works")
    st.image("https://scikit-learn.org/stable/_images/grid_search_workflow.png", caption="GridSearchCV Workflow", width=600)
    st.info("The system was trained on a balanced mental health dataset using both Logistic Regression and Linear SVM. The best performing model was selected.")

# -----------------------
# Interpretability Page
# -----------------------
elif page == "🧠 Interpretability":
    st.title("🧠 Model Interpretability")
    st.write("Understand which words are most important for each classification.")
    
    if interpret_data:
        selected_class = st.selectbox("Select Category", list(interpret_data.keys()))
        top_words = interpret_data[selected_class]
        
        st.write(f"Top 10 most influential words for **{selected_class}**:")
        
        # Display as a horizontal bar chart or just tags
        cols = st.columns(5)
        for i, word in enumerate(top_words):
            cols[i % 5].markdown(f"""<div style="background-color:#f0f2f6; color:#31333f; padding:10px; border-radius:5px; text-align:center; border-left:5px solid {class_colors.get(selected_class, 'gray')}"><b>{word}</b></div>""", unsafe_allow_html=True)
            
        st.markdown("---")
        st.write("These features were extracted from the Logistic Regression coefficients, showing the words that drive the model towards a specific prediction.")
    else:
        st.warning("Interpretability data not available. Please ensure Logistic Regression was part of the training.")

# -----------------------
# About Page
# -----------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About MoodMatrix")
    st.markdown("""
    ### 👋 Mission
    MoodMatrix is an NLP-driven tool designed to help identify potential mental health states from text input. 
    It is intended as a **demonstration of NLP capabilities** and NOT as a medical diagnostic tool.
    
    ### 🛠️ Technical Stack
    - **Preprocessing**: Custom cleaning with NLTK lemmatization.
    - **Vectorization**: TF-IDF with N-gram (1,2) support.
    - **Models**: Scikit-Learn Pipeline (Logistic Regression / Linear SVC).
    - **Optimization**: GridSearchCV for hyperparameter tuning.
    - **Frontend**: Streamlit.
    
    ### 📁 Project Structure
    ```
    MoodMatrix/
    ├── mood_app.py       # Streamlit UI
    ├── trainer.py        # ML Training Pipeline
    ├── preprocessing.py  # NLP Preprocessing
    ├── data/             # Local Dataset
    └── model/            # Saved Models & Data
    ```
    """)

# -----------------------
# Footer
# -----------------------
st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")
