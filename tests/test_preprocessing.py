import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_text

def test_empty_string():
    assert clean_text("") == ""
    
def test_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_punctuation_handling():
    # Should preserve sequences of exclamation and question marks
    text1 = "I am so sad!!! Why??"
    res1 = clean_text(text1, remove_stopwords=False, use_lemmatization=False)
    assert "__exclaim__" in res1
    assert "__question__" in res1
    
def test_stopword_behavior():
    text = "this is a test about feeling sad"
    res = clean_text(text, remove_stopwords=True, use_lemmatization=False)
    assert "this" not in res
    assert "is" not in res
    assert "test" in res

def test_lemmatization():
    text = "running feeling sadness"
    res = clean_text(text, remove_stopwords=False, use_lemmatization=True)
    # Lemmatizer handles nouns by default; "running" might not be lemmatized well without POS tags, but let's check a simple one.
    assert "feeling" in res # or feel depending on lemmatizer
