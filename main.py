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
from sklearn.model_selection import train_test_split

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Embedding, SimpleRNN
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer

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
# print(data.head())
# print(data.info())

data.columns = data.columns.str.strip()
print(data.columns)

#identifyin different polarities
# print(data['polarity of tweetï¿½'].value_counts())

#removing unwanted columns
data.drop(columns=['user', 'query', 'date of the tweet', 'id of the tweet'], inplace=True)
# print(data.head())

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

#TODO Model-Selection
#TODO Hyperparameter-Tuning
#TODO Cross-validation
#TODO Model-Interpretability
#TODO Evaluation-Metrics

#EDA on pre-processed dataset

#polarity distribution
# plt.pie(data['polarity of tweetï¿½'].value_counts(), labels=['negative', 'positive'])
# plt.show()

# #word-cloud
# def show_wordcloud(data):
#     wordcloud = WordCloud(
#         background_color='white',
#         stopwords=swords,
#         max_words=100,
#         max_font_size=30,
#         scale=3,
#         random_state=1)

#     wordcloud=wordcloud.generate(str(data))

#     fig = plt.figure(1, figsize=(12, 12))
#     plt.axis('off')

#     plt.imshow(wordcloud)
#     plt.show()

# show_wordcloud(data['pre-processed-data'])

# freq = {}
# for doc in data['pre-processed-data']:
#     for word in doc:
#         freq[word] = freq.get(word, 0) + 1

# plt.figure(figsize=(10,6))
# plt.hist(freq.values(), bins=50, color='orange')
# plt.title('Histogram of Word Frequencies')
# plt.xlabel('Word Frequency')
# plt.ylabel('Number of Words')
# plt.show()

# #pos_tagging
# def posTag(tokens):
#     tagWords = pos_tag(tokens)
#     return tagWords

# data['pos-tags'] = data['pre-processed-data'].apply(posTag)
# tags = [tag for tags in data['pos-tags'] for _, tag in tags]
# tagDis = nltk.FreqDist(tags)

# plt.figure(figsize=(10,6))
# tagDis.plot(cumulative=False)
# plt.title('Distribution of POS Tag')
# plt.xlabel('Pos Tag')
# plt.ylabel('Frequency')
# plt.show()

#model-selection

texts = data['pre-processed-data'].apply(lambda x: str(x))

#LSTM
# Test Loss: -1771.4959716796875
# Test Accuracy: 0.5195925831794739
# model = Sequential()
# tokenizer = Tokenizer()

# tokenizer.fit_on_texts(texts)
# seq = tokenizer.texts_to_sequences(texts)
# maxSeq = 100
# paddedSeq = pad_sequences(seq, maxSeq)

# model.add(Embedding((len(tokenizer.word_index)+1), 100, input_length=maxSeq))
# model.add(LSTM(128))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# XTrain, XTest, yTrain, yTest = train_test_split(paddedSeq, data['polarity of tweetï¿½'], test_size=0.35, random_state=42)
# history = model.fit(XTrain, yTrain, epochs=5, batch_size=50, validation_split=0.3)

# loss, accuracy = model.evaluate(XTest, yTest)
# print(f'Test Loss: {loss}')
# print(f'Test Accuracy: {accuracy}')

#Simple RNN
# Test Loss: 0.20048177242279053
# Test Accuracy: 0.0
model = Sequential()
tokenizer = Tokenizer() 

tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)
maxSeq = 100
paddedSeq = pad_sequences(seq, maxlen=maxSeq, padding='post')

vocabSize = len(tokenizer.word_index) + 1

model.add(Embedding(input_dim=vocabSize, output_dim=100, input_length=maxSeq))
model.add(SimpleRNN(128, return_sequences=True))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

XTrain, XTest, yTrain, yTest = train_test_split(paddedSeq, data['polarity of tweetï¿½'], train_size=0.35, random_state=42)

history = model.fit(XTrain, yTrain, epochs=5, batch_size=32, validation_split=0.3)

loss, accuracy = model.evaluate(XTest, yTest)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')