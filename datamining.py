import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import joblib

columns = ['id','account','Label','Text']

#Training data
df = pd.read_csv("data/twitter_training.csv", names= columns)
df.dropna(inplace=True) #remove N/A data points
df.isna().sum() #check if N/A exists
df.drop_duplicates(inplace=True)    #remove duplicates
df.info() #prints - ensure the numbers of entities in each column
label_count=df['Label'].value_counts()

print(df['Label'])

print("\n",label_count,"\n")

#Test data
"""
df_test = pd.read_csv("data/twitter_validation.csv", names= columns)
df_test.dropna(inplace=True)
df_test.isna().sum()
df_test.drop_duplicates(inplace=True)
df_test.info() #prints
label_count_test=df_test['Label'].value_counts()
print("\n",label_count_test,"\n")
"""

nlp=spacy.load('en_core_web_sm')

def preprocess(txt):
    if pd.isna(txt):  # Check for missing values
        return ""

    doc = nlp(txt)

    filtered_text = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_text.append(token.lemma_)

    return ' '.join(filtered_text)

df['updated text']=df['Text'].apply(preprocess)
#df_test['updated text']=df_test['Text'].apply(preprocess)

encode=LabelEncoder()
df['Label_transform']=encode.fit_transform(df['Label'])

#df_test['Label_transform']=encode.transform(df_test['Label'])

X=df['updated text']
y=df['Label_transform']
print("THIS IS IT")
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

"""
X_train=df['updated text']
y_train=df['Label_transform']
X_test=df_test['updated text']
y_test=df_test['Label_transform']
"""

v=TfidfVectorizer(ngram_range=(1, 2))
X_train_cv=v.fit_transform(X_train)
X_test_cv=v.transform(X_test)

joblib.dump(X_train_cv, 'pickles/X_train_cv.pkl')
joblib.dump(X_test_cv, 'pickles/X_test_cv.pkl')
joblib.dump(y_train, 'pickles/y_train.pkl')
joblib.dump(y_test, 'pickles/y_test.pkl')
joblib.dump(v, 'pickles/tfidf_vectorizer.pkl')

print("All done :)\nVariables ready to be used elsewhere")
