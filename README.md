# Sentiment Analysis on Twitter Data

This project focuses on sentiment analysis of Twitter data using various machine learning and deep learning techniques. It covers data preprocessing, model development, hyperparameter optimization, model evaluation, and interpretability.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Development](#model-development)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Model Evaluation](#model-evaluation)
- [Model Interpretability](#model-interpretability)
- [Results](#results)

## Overview
This project performs sentiment analysis on a dataset of tweets. The dataset contains information about tweets and their corresponding sentiments. The goal is to classify the sentiment of each tweet into one of the categories: neutral, positive, or negative.

## Installation and Requirements
1. **Python** 3.12.x
2. **Requirements** For all the required libraries and modules refer to the requirements.txt
file or just type this command in your IDE terminal. Make sure to locate this file first
    ``` pip install -r requirements.txt ```

## Data Processing
1. Download the dataset from this <a href="https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data?select=train.csv">link</a> and store it a folder name **data**
2. The pre-processing steps include:
- **Loading the Data** Train and test are loaded from CSV Files
- **Cleaning the Data** Removing unnecessary columns, lowercasing of texts, removing special character, URLs and stop words
- **Tokenization and Lemmetization** Text is tokenized the lemmatized for better analysis
    ```python
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
    
    return ' '.join(tokens) ```