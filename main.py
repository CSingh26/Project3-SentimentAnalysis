#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ssl
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
from nltk.probability import FreqDist

import tensorflow as tf

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.optimizers import Adam, RMSprop, SGD
from keras._tf_keras.keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

import keras_tuner as kt

import warnings
warnings.filterwarnings('ignore')

swords = set(STOPWORDS)

#downloading nltk packages
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')
# nltk.download("wordnet")
# nltk.download("omw-1.4")

from nltk.corpus import stopwords

#TODO Evaluation-Metrics

#loading trainData

#train-trainData
trainData = pd.read_csv('data/train.csv', encoding='latin1')

#test-trainData
testData = pd.read_csv('data/test.csv', encoding='latin1')

#cleaning data
trainData.drop(columns=['textID','Time of Tweet', 'Age of User', 'Country', 
                        'Population -2020', 'Land Area (Km²)', 
                        'Density (P/Km²)'], inplace=True)

testData.drop(columns=['textID','Time of Tweet', 'Age of User', 'Country', 
                        'Population -2020', 'Land Area (Km²)', 
                        'Density (P/Km²)'], inplace=True)

# Pre-processing function
def preprocessText(text):
    if not isinstance(text, str):
        return text
    
    #lowercasing
    text = text.lower()
    
    #removing special characters and punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    #removing URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Removing stop-words
    stop_words = set(stopwords.words("english"))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    #tokenization
    tokens = word_tokenize(text)
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

#pre-processing individual datasets 
trainData['cleanText'] = trainData['text'].apply(preprocessText)
testData['cleanText'] = testData['text'].apply(preprocessText)

#combing and pre-processing datasets
data = pd.concat([trainData, testData])
data.dropna(subset=['text', 'sentiment'], inplace=True)
data['cleanText'] = data['text'].apply(preprocessText)
data['selected_text'] = data['selected_text'].apply(preprocessText)
data.dropna(subset=['cleanText'], inplace=True)

#EDA (train data only)

#sentiment value counts
# sentimentCounts = trainData['sentiment'].value_counts(normalize=True)

# plt.figure(figsize=(10, 6))
# plt.bar(sentimentCounts.index, sentimentCounts.values)
# plt.xlabel('Sentiment')
# plt.ylabel('Proportion')
# plt.show()

# #Sentiment Histplot
# sns.histplot(trainData['sentiment'], kde=True, color='c')
# plt.show()

# #Word Frequency Distribution
# wordFreq = FreqDist(word_tokenize(' '.join(trainData['sentiment'])))
# plt.figure(figsize=(10, 6))
# wordFreq.plot(20, cumulative=False)
# plt.title("Word Frequency Distribution")
# plt.xlabel('Word')
# plt.ylabel('Frequency')
# plt.show()

# #WordCloud
# text_data = ' '.join(trainData['cleanText'].dropna().astype(str))
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off') 
# plt.title('Word Cloud for Training Data')
# plt.show()

#Model-Selection

X = data['cleanText']
y = data['sentiment']

#Spliting Data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

#TF-IDF
vec = TfidfVectorizer(max_features=5000)
XVTrain = vec.fit_transform(XTrain)
XVTest = vec.transform(XTest)

score = data['sentiment'].value_counts(normalize=True).max()
print(score)

yTrainEnc = pd.get_dummies(yTrain).values
yTestEnc = pd.get_dummies(yTest).values

#MultiLayer Perception (MLP) model with Adam Optimizer
# model = Sequential()
# model.add(Dense(128, input_dim=XVTrain.shape[1], activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(yTrainEnc.shape[1], activation='softmax'))

# opt = Adam(learning_rate=0.0001)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# history = model.fit(XVTrain.toarray(), yTrainEnc, epochs=20, batch_size=2, validation_data=(XVTest.toarray(), yTestEnc), verbose=1)

# loss, accuracy = model.evaluate(XVTest.toarray(), yTestEnc, verbose=1)
# print(f'Test Accuracy: {accuracy:.4f}')
# print(f'Test Loss: {loss:.4f}')

#MultiLayer Perception (MLP) model with RMSprop Optimizer
# model = Sequential()
# model.add(Dense(128, input_dim=XVTrain.shape[1], activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(yTrainEnc.shape[1], activation='softmax'))

# opt = RMSprop(learning_rate=0.0001)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# history = model.fit(XVTrain.toarray(), yTrainEnc, epochs=5, batch_size=2, validation_data=(XVTest.toarray(), yTestEnc), verbose=1)

# print("Before HyperParameter Optimization")
# loss, accuracy = model.evaluate(XVTest.toarray(), yTestEnc, verbose=1)
# print(f'Test Accuracy: {accuracy:.4f}')
# print(f'Test Loss: {loss:.4f}')

#Stats for Adam Optimizer Model 
# Test Accuracy: 0.6357
# Test Loss: 2.1005

#Stats for RMSprop Optimizer Model
# Test Accuracy: 0.7116
# Test Loss: 0.8713

#Going Forward with the RMSprop Optimizer Model because of 
#1 Less Training Time (24s/step)
#2 Higher Accuracy
#3 Lower Loss

#hyperparameter optimzation
def buildModel(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units_input', min_value=32, max_value=512, step=32),
        input_dim=XVTrain.shape[1],
        activation=hp.Choice('activation_input', values=['relu', 'tanh', 'sigmoid'])
    ))
    model.add(Dense(
        units=hp.Int('units_hidden', min_value=32, max_value=512, step=32),
        activation=hp.Choice('activation_hidden', values=['relu', 'tanh', 'sigmoid'])
    ))
    model.add(Dense(yTrainEnc.shape[1], activation='softmax'))
    
    optimizer = hp.Choice('optimizer', values=['rmsprop', 'adam', 'sgd'])
    if optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    elif optimizer == 'adam':
        opt = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    else:
        opt = SGD(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))

    loss = hp.Choice('loss', values=['categorical_crossentropy', 'mean_squared_error'])
    
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    return model

#Cross-Validation

n = 5
XV = vec.fit_transform(X)
skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)

acc = []
losses = []

label_encoder = LabelEncoder()
yEnc = label_encoder.fit_transform(y)
yEnc = to_categorical(yEnc, num_classes=len(label_encoder.classes_))

for i, j in skf.split(XV, np.argmax(yEnc, axis=1)):
    XTrain, XTest, = XV[i], XV[j]
    yTrain, yTest = yEnc[i], yEnc[j]

    XTrain = XTrain.toarray()
    XTest = XTest.toarray()

    #setting up tuner
    tuner = kt.GridSearch(
        buildModel,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='nlpModelTunning'
    )

    #finding the best possible hyperparameter
    tuner.search(XTrain, yTrain, epochs=10, validation_data=(XTest, yTest))

    #stroing the params
    bestParas = tuner.get_best_hyperparameters(num_trials=1)[0]

    #building and fiting the model with the params
    model = tuner.hypermodel.build(bestParas)
    model.fit(XTrain, yTrain, epochs=6, batch_size=2, 
              validation_data=(XTest, yTest), verbose=1)
    
    #evaluating losses and accuracies 
    loss, accuracy = model.evaluate(XTest, yTest, verbose=1)
    acc.append(accuracy)
    losses.append(loss)

#taking the averages
avgAcc = np.mean(acc)
avgLoss = np.mean(losses)

print('After HyperParameter Optimization and Cross-validation')
print(f'Average Test Accuracy: {avgAcc:.4f}')
print(f'Average Test Loss: {avgLoss:.4f}')

#Model-Interpretability

#Class for preicitng probalities
class PredictWrapper:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vec = vectorizer

    def predict_proba(self, texts):
        vecText = self.vec.transform(texts).toarray()
        return self.model.predict(vecText)
        

pred = PredictWrapper(model, vec)

#LIME explainer
explainer = lime.lime_text.LimeTextExplainer(class_names=['neutral', 'positive', 'negative'])

#ensuring XTest is a list of strings
XTest = XTest if isinstance(XTest, list) else XTest.tolist()
XTest = [str(text) for text in XTest]

#instanceto explain
idx = 15
textInstance = XTest[idx]

#ensuring the instance is a single string
if isinstance(textInstance, list):
    textInstance = ' '.join(textInstance)
if isinstance(textInstance, bytes):
    textInstance = textInstance.decode('utf-8')

#Getting explaination 
exp = explainer.explain_instance(textInstance, pred.predict_proba)

#Visualizing the explaination
try:
    exp.show_in_notebook(text=True)
except ModuleNotFoundError:
    print("IPython is not installed. Skipping notebook display.")

#saving into a html file
exp.save_to_file('explaination.html')
