import pandas as pd
import joblib

df_emoji = pd.read_csv("data/Emoji_Sentiment_Data.csv", usecols=['Emoji', 'Negative', 'Neutral', 'Positive'])
df_emoji['Sentiment'] = df_emoji.apply(
    lambda row:
    1 if (row['Negative'] > row['Positive'] and row['Negative'] > row['Neutral'])
    else 2 if (row['Neutral'] > row['Negative'] and row['Neutral'] > row['Positive'])
    else 3 if (row['Positive'] > row['Negative'] and row['Positive'] > row['Neutral'])
    else 0, axis=1)

'''
if res == 0: return "Irrelevant"
if res == 1: return "Negative"
if res == 2: return "Neutral"
if res == 3: return "Positive"
'''

# Create emoji sentiment dictionary for faster lookups
emoji_sentiment_dict = dict(zip(df_emoji['Emoji'], df_emoji['Sentiment']))

joblib.dump(emoji_sentiment_dict, 'pickles/emoji_sentiment_dict.pkl')