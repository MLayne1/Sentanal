import os
import re
import json
import csv
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
	# Open files for reading
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

	# Return the arrays
	return pos, neg

# Allows user to set seed for shuffle
def seedAndShuffle(seed, real, fake):
	# Set Random's seed if desired
	# random.seed(seed)
	# print("\nUsing seed: " + str(seed))
	# Shuffle the articles randomly
	return random.shuffle(real), random.shuffle(fake)


# splits training and testing data based on a parameter
def setSplit(split, real, fake):
	# train using split as %
	split = int(split*len(fake))
	# Separate lists 
	trainReal = real[:split]
	trainFake = fake[:split]
	testReal = real[(split+1):]
	testFake = fake[(split+1):]

	# create training and testing list
	train = trainReal+trainFake
	test = testReal+testFake

	# Print info on split
	print("\nLength of training set: = {0} real + {1} fake = {2}".format(len(trainReal),len(trainFake),len(train)))
	print("Length of test set: = {0} real + {1} fake = {2}".format(len(testReal),len(testFake),len(test)) + "\n")

	return train, test

# Runs the NLTK Sentiment Analyzer
def runSentanal(train, test):
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

	# creates array for storing values
	values = []

	# display results
	for key,value in sorted(sentanal.evaluate(testList).items()):
		print('{0}: {1}'.format(key, value))
		values.append(value)

	# write results to csv
	with open('Data\\sentanalResults.csv', mode='a') as csvFile:
		writer = csv.writer(csvFile, delimiter=',')
		writer.writerow(values)

# Main with seed as parameter
def mainRunner(seed):
	real, fake = generateArrays()
	seedAndShuffle(seed, real, fake) # Original: 9245
	train, test = setSplit(0.3, real, fake) # first param is % to train with
	runSentanal(train, test)

# main without seed parameter
def main():
	mainRunner(9245)

def generateData(numOfRuns):
	for x in range(0, numOfRuns):
	    print("\n\nRunning attempt {0} of {1}".format(x+1, numOfRuns))	
	    main()

generateData(3)