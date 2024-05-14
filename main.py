#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import ssl
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from nltk import pos_tag
swords = set(STOPWORDS)

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
print(data['polarity of tweetï¿½'].value_counts())

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

#EDA on pre-processed dataset

#polarity distribution
plt.pie(data['polarity of tweetï¿½'].value_counts(), labels=['negative', 'positive'])
plt.show()

#word-cloud
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=swords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(data['pre-processed-data'])

freq = {}
for doc in data['pre-processed-data']:
    for word in doc:
        freq[word] = freq.get(word, 0) + 1

plt.figure(figsize=(10,6))
plt.hist(freq.values(), bins=50, color='orange')
plt.title('Histogram of Word Frequencies')
plt.xlabel('Word Frequency')
plt.ylabel('Number of Words')
plt.show()

#pos_tagging
def posTag(tokens):
    tagWords = pos_tag(tokens)
    return tagWords

data['pos-tags'] = data['pre-processed-data'].apply(posTag)
tags = [tag for tags in data['pos-tags'] for _, tag in tags]
tagDis = nltk.FreqDist(tags)

plt.figure(figsize=(10,6))
tagDis.plot(cumulative=False)
plt.title('Distribution of POS Tag')
plt.xlabel('Pos Tag')
plt.ylabel('Frequency')
plt.show()