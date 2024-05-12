#imports 
import pandas as pd
import numpy as np
import matplotlib as plt
import nltk
import ssl
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#downloading nltk packages
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords

#loading the data
data = pd.read_csv('data/data.csv', encoding='iso-8859-1')

#viewing the data
print(data.head())
print(data.info())

data.columns = data.columns.str.strip()
print(data.columns)

#identifyin different polarities
print(data['polarity of tweet'].value_counts())

#removing unwanted columns
data.drop(columns=['user', 'query', 'date of the tweet', 'id of the tweet'], inplace=True)
print(data.head())

#Data Pre-prcocessing

#lowercasing 
data['pre-processed-data'] = data['text of the tweetï¿½'].apply(lambda
                            x: x.lower() if isinstance(x, str) else x)

#removing puncuation marks and special characters
punch = r'[^\w\s]'

data['pre-processed-data'] = data['pre-processed-data'].apply(lambda
                            x: re.sub(punch, '', x) if isinstance(x, str) else x)

#removing urls (if present)
def removeURL(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

data['pre-processed-data'] = data['pre-processed-data'].apply(lambda
                            x: removeURL(x) if isinstance(x, str) else x)

#removing stop-words
def stepWordRemoval(text):
    nltk
    stwd = set(stopwords.words("english"))
    wdtkns = text.split()
    fil = [word for word in wdtkns if word not in stwd]
    return ' '.join(fil)

data['pre-processed-data'] = data['pre-processed-data'].apply(lambda
                            x: stepWordRemoval(x) if isinstance(x, str) else x)

#tokenization
vec = TfidfVectorizer()
tknMatix = vec.fit_transform(data['pre-processed-data'].values.astype('U'))

data['pre-processed-data'] = vec.inverse_transform(tknMatix)

#lemmatization
lem = WordNetLemmatizer()

data['pre-processed-data'] = data['pre-processed-data'].apply(lambda
                            x: lem.lemmatize(x) if isinstance(x, str) else x)
