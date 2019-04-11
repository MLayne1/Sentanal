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

SRC_TRAIN = ".\\Data\\jsonFiles\\train.json"
SRC_TEST = ".\\Data\\jsonFiles\\test.json"

nFalsePositive = 0
nTruePositive = 0

nFalseNegative = 0
nTrueNegative = 0

def measure(givenLabel, trueLabel):
    if givenLabel == trueLabel:
        if trueLabel == 'pos':
            global nTruePositive
            nTruePositive += 1
            return True
        else:
            global nTrueNegative
            nTrueNegative += 1
            return True
    else:
        if trueLabel == 'neg':
            global nFalsePositive
            nFalsePositive += 1
            return False
        else:
            global nFalseNegative
            nFalseNegative += 1
            return False

def main():
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

            givenLabel = cl.classify(article['text'])
            trueLabel = article['label']

            correct = measure(givenLabel, trueLabel)

            print(str(count) + " Classified:" + givenLabel + " Label:" + trueLabel + (" correct" if correct else " wrong"))


    accuracy = (nTruePositive + nTrueNegative) / (nTruePositive + nTrueNegative + nFalsePositive + nFalseNegative)
    fMeasure = (2 * nTruePositive) / ((2 * nTruePositive) + nFalsePositive + nFalseNegative)

    print("accuracy: {0}".format(accuracy))
    print("F1-Score: {0}".format(fMeasure))

    print("TP: {0} FP: {1} TN: {2} FN: {3}".format(nTruePositive, nFalsePositive, nTrueNegative, nFalseNegative))

main()