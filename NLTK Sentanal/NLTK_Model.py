import os
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

"""
This is a sentiment analyses model using the NLTK classifier

Authors: Luis Hernandez, Jordan Jefferson, Matthew Layne
"""

trainFile = open('.\\Data\\jsonFiles\\train.json') 
testFile = open('.\\Data\\jsonFiles\\test.json') 

trainArticlesJson = json.load(trainFile)

trainReal = []
trainFake = []


for article in trainArticlesJson:

	if article['label'] == 'pos':
		trainReal.append(article['text'])
	else:
		trainFake.append(article['text'])

print("Length of real training set: " + str(len(trainReal)))
print(len(trainFake))




# FakeArticlesJson = json.load()
# RealArticlesJson = json.load()