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

# Generates arrays of positive and degative labelled articles
def generateArrays():
	trainFile = open('.\\Data\\jsonFiles\\train.json') 
	testFile = open('.\\Data\\jsonFiles\\test.json') 
	trainArticlesJson = json.load(trainFile)
	testArticlesJson = json.load(testFile)

	# Create empty arrays to store articles in array format
	pos = []
	neg = []

	# Append pos and neg labelled articles to the appropriate array
	for article in trainArticlesJson:
		wordTokens = word_tokenize(article['text'])
		label = article['label']
		tup = (wordTokens, label)
		if article['label'] == 'pos':
			pos.append(tup)
		else:
			neg.append(tup)
	for article in testArticlesJson:
		wordTokens = word_tokenize(article['text'])
		label = article['label']
		tup = (wordTokens, label)
		if article['label'] == 'pos':
			pos.append(tup)
		else:
			neg.append(tup)
	return pos, neg
	

real, fake = generateArrays()


# Seed Random if desired
random.seed(9245)
# Shuffle the articles randomly
random.shuffle(real)
random.shuffle(fake)

# validate appropriate values
# this was used to find my dumb parsing mistake
# for doc in real:
# 	print(type(doc))		# print tuple
# 	print(type(doc[0]))	# print wordTokens
# 	print(doc[1])	# print label


# Separate lists 
trainReal = real[:50]
trainFake = fake[:50]
testReal = real[51:]
testFake = fake[51:]

# create training and testing list
train = trainReal+trainFake
test = testReal+testFake

# print(type(train))
# print(type(train[0]))
# print(type(train[0][0]))
# print(type(train[0][1]))

sentanal = SentimentAnalyzer()

all_words_neg = sentanal.all_words([mark_negation(doc) for doc in train])
unigramFeats = sentanal.unigram_word_feats(all_words_neg, min_freq=4)
# print(len(unigramFeats))
# print(unigram_feats)
sentanal.add_feat_extractor(extract_unigram_feats, unigrams=unigramFeats)


trainList = sentanal.apply_features(train)
testList = sentanal.apply_features(test)

trainer = NaiveBayesClassifier.train
classifier = sentanal.train(trainer, trainList)

# display results
for key,value in sorted(sentanal.evaluate(testList).items()):
	print('{0}: {1}'.format(key, value))

print("Length of training set: = {0} real + {1} fake = {2}".format(len(trainReal),len(trainFake),len(train)))
print("Length of test set: = {0} real + {1} fake = {2}".format(len(testReal),len(testFake),len(test)))