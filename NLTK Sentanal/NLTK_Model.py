import os
import re
import json
import nltk
import random
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
testArticlesJson = json.load(testFile)

# Create empty arrays to store articles in array format
real = []
fake = []

# Append real and fake articles to the appropriate array
for article in trainArticlesJson:
	if article['label'] == 'pos':
		real.append(article['text'])
	else:
		fake.append(article['text'])

for article in testArticlesJson:
	if article['label'] == 'pos':
		real.append(article['text'])
	else:
		fake.append(article['text'])

# Seed Random if desired
random.seed(9245)
# Shuffle the articles randomly
random.shuffle(real)
random.shuffle(fake)

# Verify Proper seeding
# testArr = [1,2,3,4,5,6,7,8,9]
# print(testArr)
# random.shuffle(testArr)
# print(testArr)


# print("Length of real training set: " + str(len(trainReal)))
# print("Length of real test set: " + str(len(testReal)))
# print("Length of fake training set: " + str(len(trainFake)))
# print("Length of fake test set: " + str(len(testFake)))




# FakeArticlesJson = json.load()
# RealArticlesJson = json.load()