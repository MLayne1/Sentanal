from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import json

print("Running!")


# train textblob NaiveBayesClassifier
with open('train.json', encoding='utf-8', mode='r') as train:
    cl = NaiveBayesClassifier(train, format="json")
    cl.show_informative_features(10)

# classify each 
with open('test.json', encoding='utf-8') as test:
    articles = json.load(test)

    print("to classify: " + str(len(articles)) )

    count = 0

    for article in articles:
        count+=1
        correct = str(cl.classify(article['text'])) == article['label']
        print(str(count) + " C:" + str(cl.classify(article['text'])) + " Label:" + article['label'] + (" correct" if correct else " wrong"))


with open('test.json', encoding='utf-8') as x:
    print("accuracy: " + str(cl.accuracy(x, format="json")))

print("Done!!")

# prob_dist = cl.prob_classify("This one's a doozy.")

# v = prob_dist.max()

# print(v)

# v = round(prob_dist.prob("pos"), 2)
# print(v)

# v = round(prob_dist.prob("neg"), 2)
# print(v)
