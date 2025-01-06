import joblib
import spacy
import emoji

nlp = spacy.load('en_core_web_sm')
v = joblib.load('pickles/tfidf_vectorizer.pkl')
model = joblib.load('pickles/svm_model.pkl')
emoji_sentiment_dict = joblib.load('pickles/emoji_sentiment_dict.pkl')


def preprocess(txt):
    doc = nlp(txt)
    filtered_text = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_text.append(token.lemma_)
    return ' '.join(filtered_text)


def decode_emoji(res):
    if res <= 0: return "Irrelevant"
    if res <= 1.49: return "Negative"
    if res <= 2.2: return "Neutral"
    if res <= 3: return "Positive"


def decode(res):
    if res == 0: return "Irrelevant"
    if res == 1: return "Negative"
    if res == 2: return "Neutral"
    if res == 3: return "Positive"


def analyze_emoji(emojis):
    emoji_sentiments = [emoji_sentiment_dict[e] for e in emojis
                        if e in emoji_sentiment_dict]

    emoji_sentiment_score = (sum(emoji_sentiments) / len(emoji_sentiments)
                             if emoji_sentiments else 2)

    emoji_sentiment = decode_emoji(emoji_sentiment_score)
    return emoji_sentiment_score, emoji_sentiment


def analyze(u_input):
    emoji_list = [c for string in u_input for c in string if emoji.is_emoji(c)]
    text_input = [''.join(c for string in u_input for c in string if not emoji.is_emoji(c))]

    if not text_input[0].strip() == '':
        processed_input = [preprocess(text_input[0])]
        user_input_cv = v.transform(processed_input)

        encoded_text_sentiment = model.predict(user_input_cv)
        text_sentiment = decode(encoded_text_sentiment)
        print(f"Text sentiment: {encoded_text_sentiment[0]} {text_sentiment}")

    if emoji_list:
        emoji_sentiment_score, emoji_sentiment = analyze_emoji(emoji_list)
        print(f"Emoji sentiment: {emoji_sentiment_score} {emoji_sentiment}")

    if (not text_input[0].strip() == '') and emoji_list:
        combined_sentiment = (encoded_text_sentiment + emoji_sentiment_score) / 2

        if combined_sentiment < 1.7:
            final_sentiment = "Negative"
        elif combined_sentiment < 2.1:
            final_sentiment = "Neutral"
        else:
            final_sentiment = "Positive"

        if (emoji_sentiment == 'Negative' and text_sentiment == 'Positive') or (
                emoji_sentiment == 'Positive' and text_sentiment == 'Negative'):
            final_sentiment += ", might be sarcasm! 😏"

        print(f"Combined sentiment: {final_sentiment}")
        return final_sentiment
    elif emoji_list:
        return emoji_sentiment
    else:
        return text_sentiment


def talk():
    while True:
        user_input = [input("\nWhat would you like to analyze?\n")]
        if user_input[0].strip().lower() == "bye":
            print("Good bye 😊")
            exit()

        if user_input[0].strip():  # Check if input is not empty, or just spaces, as "  "
            break
        print("You didn't enter anything 😬\nTry again.")

    analyze(user_input)
    talk()

# Uncomment this to run SVC through terminal
# print("Hello darling ✨")
# talk()
