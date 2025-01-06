import pandas as pd
import joblib

#Create the dictionary
df_emoji = pd.read_csv("data/Emoji_Sentiment_Data.csv", usecols=['Emoji', 'Negative', 'Neutral', 'Positive'])
df_emoji['Sentiment_score'] = df_emoji.apply(
    lambda row:
    1 if (row['Negative'] > row['Positive'] and row['Negative'] > row['Neutral'])
    else 2 if (row['Neutral'] > row['Negative'] and row['Neutral'] > row['Positive'])
    else 3 if (row['Positive'] > row['Negative'] and row['Positive'] > row['Neutral'])
    else 0, axis=1)

emoji_sentiment_dict = dict(zip(df_emoji['Emoji'], df_emoji['Sentiment_score']))
joblib.dump(emoji_sentiment_dict, 'pickles/emoji_sentiment_dict.pkl')

#Cretate csv for visual overview
df_emoji['Sentiment'] = df_emoji.apply(
    lambda row:
    'Negative' if (row['Sentiment_score'] == 1)
    else 'Neutral' if (row['Sentiment_score'] == 2)
    else 'Positive' if (row['Sentiment_score'] == 3)
    else 'Irrelevant', axis=1)

df_filtered = df_emoji.iloc[:, [0, 4, 5]]
df_filtered.to_csv('data/emoji_sentiments.csv', index=False)

print("Emoji dictionary ready to be used elsewhere")