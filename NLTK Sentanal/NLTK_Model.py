import os
import re
import json
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

"""
This is a sentiment analyses model using the NLTK classifier

Authors: Luis Hernandez, Jordan Jefferson, Matthew Layne
"""

trainFile = open('.\\Data\\jsonFiles\\train.json') 
testFile = open('.\\Data\\jsonFiles\\test.json') 

trainArticlesJson = json.load(trainFile)
testArticlesJson = json.load(testFile)

# Create empty arrays to store tuples in format ([wordTokens], 'label')
real = []
fake = []

# Append real and fake articles to the appropriate array
for article in trainArticlesJson:
	wordTokens = word_tokenize(article['text'])
	label = article['label']
	tup = (wordTokens, label)
	if article['label'] == 'pos':
		real.append(tup)
	else:
		fake.append(tup)

for article in testArticlesJson:
	wordTokens = word_tokenize(article['text'])
	label = article['label']
	tup = (wordTokens, label)
	if article['label'] == 'pos':
		real.append(article['text'])
	else:
		fake.append(article['text'])

# Seed Random if desired
random.seed(9245)
# Shuffle the articles randomly
random.shuffle(real)
random.shuffle(fake)

# Choose set for training and testing
trainReal = real[:50]
trainFake = fake[:50]
testReal = real[51:]
testFake = fake[51:]

print(len(real))
print("Length of real training set: {0}".format(len(trainReal)))
print("Length of real test set: {0}".format(len(testReal)))
print("Length of fake training set: {0}".format(len(trainFake)))
print("Length of fake test set: {0}".format(len(testFake)))

# Verify Proper seeding
# testArr = [1,2,3,4,5,6,7,8,9]
# print(testArr)
# random.shuffle(testArr)
# print(testArr)




# FakeArticlesJson = json.load()
# RealArticlesJson = json.load()