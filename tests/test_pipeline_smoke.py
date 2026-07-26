import pytest
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pipeline_smoke():
    # Load model
    model = joblib.load("model/moodmatrix_model.joblib")
    
    # Sample sentences
    sentences = [
        "I feel so great today, everything is going well!",
        "I just can't take this anymore, I want it all to end.",
        "My anxiety is keeping me up at night, I am so worried."
    ]
    
    # Predict
    preds = model.predict(sentences)
    
    assert len(preds) == 3
    # Check if predictions are strings (class names)
    for p in preds:
        assert isinstance(p, str)
        assert len(p) > 0
