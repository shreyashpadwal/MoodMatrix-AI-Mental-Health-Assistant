import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data — required for Streamlit Cloud
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)

def clean_text(text, use_lemmatization=True):
    """
    Enhanced preprocessing for mental health text classification.
    """
    if not isinstance(text, str):
        return ""

    # Lowercasing
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization (split by whitespace)
    tokens = text.split()

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatization
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    cleaned_text = " ".join(tokens)
    return cleaned_text.strip()
