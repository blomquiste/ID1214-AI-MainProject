import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the irony detection model and tokenizer
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
try:
    irony_tokenizer = AutoTokenizer.from_pretrained(IRONY_MODEL)
    irony_model = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL)
except OSError as e:
    print(f"Error loading irony model: {e}")
    exit()

# Load the sentiment analysis model and tokenizer
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
except OSError as e:
    print(f"Error loading sentiment model: {e}")
    exit()


def analyze_irony(text):
    """Analyze irony using the RoBERTa model."""
    encoded_text = irony_tokenizer(text, return_tensors='pt')
    output = irony_model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Return the scores for irony and not irony
    irony_results = {
        "not_irony": round(float(scores[0]) * 100, 2),  # Convert to percentage
        "irony": round(float(scores[1]) * 100, 2)  # Convert to percentage
    }
    return irony_results


def analyze_sentiment(text):
    """Analyze sentiment using the RoBERTa model."""
    encoded_text = sentiment_tokenizer(text, return_tensors='pt')
    output = sentiment_model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Return the scores for negative, neutral, and positive sentiment
    sentiments = {
        'Negative': float(scores[0]),
        'Neutral': float(scores[1]),
        'Positive': float(scores[2])
    }
    dominant_sentiment = max(sentiments, key=sentiments.get)
    confidence = sentiments[dominant_sentiment] * 100  # Confidence percentage

    # sentiment_results = {
    #     'dominant_sentiment': dominant_sentiment,
    #     'confidence': round(confidence, 2),
    #     'sentiments': sentiments
    # }
    return sentiments


def combined_analysis(text):
    """Combine irony detection and sentiment analysis with conditional logic."""
    # Perform irony analysis
    irony_results = analyze_irony(text)
    sentiments = analyze_sentiment(text)
    
    # If the result is not irony, we proceed to get the sentiment
    if irony_results["irony"] < 60:
        # Perform and return only sentiment analysis
        return {
            "analysis_type": "sentiment",
            "irony_analysis": irony_results,
            "sentiment_analysis": sentiments
        }
    else:
        # Perform sentiment analysis and return both
        return {
            "analysis_type": "irony",
            "irony_analysis": irony_results,
            "sentiment_analysis": sentiments
        }

# Testing
'''"that's so cool 🙄"
that's so cool!
"Today was a great day! "
"Oh no, it broke"'''
# text = "Oh no, it broke"
# print(text)
# result = combined_analysis(text)
# print(result)

# irony = "Great, it broke the first day..."
# print(irony)
# result_irony = combined_analysis(irony)
# print(result_irony)

# notirony = "Oh no, it broke"
# print(notirony)
# result_notirony = combined_analysis(notirony)
# print(result_notirony)
