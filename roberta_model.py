#----------------------------------------------------------------------------
# Using a roBERTa pretrained tranformer model
# automates the tokenizing process
# accounts for relationships between words and context
#----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the model and tokenizer once, so they don't have to be loaded each time
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment_roberta(text):
    """Analyze sentiment using the RoBERTa model."""
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Return the scores for negative, neutral, and positive sentiment
    sentiments = {
        'negative': float(scores[0]),
        'neutral': float(scores[1]),
        'positive': float(scores[2])
    }
    dominant_sentiment = max(sentiments, key=sentiments.get)
    confidence = sentiments[dominant_sentiment] * 100  # Confidence percentage
    analysis = {
        'dominant_sentiment': dominant_sentiment,
        'confidence': round(confidence, 2),
        'sentiments': sentiments
    }
    
    return analysis

# Example usage
# text = "Oh no, it broke"
# print(text)
# result = analyze_sentiment_roberta(text)
# print(result)