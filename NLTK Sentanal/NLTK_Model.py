""" This is a sentiment analysis model using NLTK and a NaiveBayesClassifier classifier """

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
from nltk.sentiment.util import mark_negation, extract_unigram_feats

__author__ = "Luis Hernandez, Jordan Jefferson, Matthew Layne"


SRC_TRAIN = '.\\Data\\jsonFiles\\train.json'
SRC_TEST = '.\\Data\\jsonFiles\\test.json'

SRC_REAL = '.\\Data\\jsonFiles\\jReal.json'
SRC_FAKE = '.\\Data\\jsonFiles\\jFake.json'

def generateTupleList(path):
	""" Given the source of a JSON file return a List of tuples

	Arguments:
		path {str} -- the path to the source JSON file
	Returns:
		{list} -- the list of tuples in format ([wordTokens], 'label')
	"""
	tupleList = []

	with open(path) as jFile:
		articlesJson = json.load(jFile)
		print(len(articlesJson))
	
	for article in articlesJson:
		wordTokens = word_tokenize(article['text'])
		label = article['label']
		tup = (wordTokens, label)
		tupleList.append(tup)

	return tupleList

def generateArrays():
	trainFile = open(SRC_TRAIN) 
	testFile = open(SRC_TEST) 
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

def seedAndShuffle(seed, toShuffle):
	# Set Random's seed if desired
	random.seed(seed)
	print("\nUsing seed: " + str(seed))
	# Shuffle the articles randomly
	return random.shuffle(toShuffle)

def setSplit(split, real, fake):
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
	sentanal.train(trainer, trainList)

	# display results
	for key,value in sorted(sentanal.evaluate(testList).items()):
		print('{0}: {1}'.format(key, value))

def main():

	# real, fake = generateArrays()

	real =  generateTupleList(SRC_REAL)
	fake =  generateTupleList(SRC_FAKE)

	seedAndShuffle(9245, real)
	seedAndShuffle(9245, fake)

	train, test = setSplit(50, real, fake)
	runSentanal(train, test)

main()