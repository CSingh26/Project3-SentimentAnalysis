#imports 
import pandas as pd
import numpy as np
import matplotlib as plt

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