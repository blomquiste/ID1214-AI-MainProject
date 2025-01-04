#----------------------------------------------------------------------------
# Using a roBERTa pretrained tranformer model
# automates the tokenizing process
# accounts for relationships between words and context
#----------------------------------------------------------------------------
import numpy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the model and tokenizer once, so they don't have to be loaded each time
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment_roberta(text):
    """Analyze sentiment using the RoBERTa model."""
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Return the scores for negative, neutral, and positive sentiment
    return {
        'negative': float(scores[0]),
        'neutral': float(scores[1]),
        'positive': float(scores[2])
    }


#--------Testing----------
# if __name__ == '__main__':
#     # Load data
#     columns = ['subject', 'label', 'text']
#     df = pd.read_csv("twitter_training.csv", names=columns)
#     example = df['text'].values[600]
    
#     # Print example text once
#     print("Analyzing text:", example)
    
#     # Load model once
#     MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
#     # Analyze sentiment
#     scores_dict = analyze_sentiment(example, tokenizer, model)
#     print("Sentiment scores:", scores_dict)

#---------Attempt at finetuning-----------------
# if __name__ == "__main__":
#     device = torch.device("mps")  # Utilize GPU if possible
#     MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#     training_args = TrainingArguments("test-trainer")  # Set training parameters for fine-tuning

#     # Load sarcasm dataset
#     train_df = pd.read_csv('./data/cleaned_dataset_train.csv', names=['tweet', 'sarcastic'], dtype={'tweet': str})
#     test_df = pd.read_csv('./data/cleaned_dataset_test.csv', names=['tweet', 'sarcastic'], dtype={'tweet': str})
#     # Split the training data into train and validation sets
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         train_df['tweet'], train_df['sarcastic'], test_size=0.2, random_state=42
#     )
#     MAX_LENGTH = 128
#     def tokenize_data(texts):
#         return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

#     train_encodings = tokenize_data(train_texts)
#     val_encodings = tokenize_data(val_texts)
#     test_encodings = tokenize_data(test_df['tweet'])

#     train_labels = train_labels.astype(int).tolist()
#     val_labels = val_labels.astype(int).tolist()
#     test_labels = test_df['sarcastic'].astype(int).tolist()

#     # Create Dataset class
#     class SarcasmDataset(torch.utils.data.Dataset):
#         def __init__(self, encodings, labels):
#             self.encodings = encodings
#             self.labels = labels

#         def __len__(self):
#             return len(self.labels)

#         def __getitem__(self, idx):
#             item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#             item['labels'] = torch.tensor(self.labels[idx])
#             return item

#     # Create datasets for trainer
#     train_dataset = SarcasmDataset(train_encodings, train_labels)
#     val_dataset = SarcasmDataset(val_encodings, val_labels)
#     test_dataset = SarcasmDataset(test_encodings, test_labels)

#     # Create Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         data_collator=DataCollatorWithPadding(tokenizer),
#         tokenizer=tokenizer
#     )

#     # Train the model
#     trainer.train()

#     # Evaluate the model
#     predictions = trainer.predict(val_encodings)
#     print(predictions.predictions.shape, predictions.label_ids.shape)

# that's so cool 🙄