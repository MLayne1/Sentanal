"""
Sentanal uses TextBlob NaiveBayesClassifier in order to detect fake news.
Used as ref: https://textblob.readthedocs.io/en/dev/classifiers.html
    "pos" = Real article
    "neg" = Fake article
"""

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import json

__authors__= "Luis Hernandez, Matthew Layne, Jordan Jefferson"
__version__ = "4.3.2019"

# SRC_TRAIN = "..\\NLTK Sentanal\\Data\\jsonFiles\\train.json"
# SRC_TEST = "..\\NLTK Sentanal\\Data\\jsonFiles\\test.json"

SRC_TRAIN = ".\\train.json"
SRC_TEST = ".\\test.json"

print("Running!")
# train textblob NaiveBayesClassifier
with open(SRC_TRAIN, encoding='utf-8', mode='r') as train:
    cl = NaiveBayesClassifier(train, format="json")
    cl.show_informative_features(10)

# classify each article in the test data
with open(SRC_TEST, encoding='utf-8') as test:

    #load json to a json object
    articles = json.load(test)
    print("to classify: " + str(len(articles)) )

    # iterate through articles
    count = 0
    for article in articles:
        count+=1
        correct = str(cl.classify(article['text'])) == article['label']
        print(str(count) + " C:" + str(cl.classify(article['text'])) + " Label:" + article['label'] + (" correct" if correct else " wrong"))

# compute accuracy
with open(SRC_TEST, encoding='utf-8') as x:
    print("accuracy: " + str(cl.accuracy(x, format="json")))

print("Done!!")



# prob_dist = cl.prob_classify("This one's a doozy.")

# v = prob_dist.max()

# print(v)

# v = round(prob_dist.prob("pos"), 2)
# print(v)

# v = round(prob_dist.prob("neg"), 2)
# print(v)
