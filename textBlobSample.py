from textblob.classifiers import NaiveBayesClassifier

cl = NaiveBayesClassifier(train)
with open('train.json', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="json")
cl.classify("This is an amazing library!")
prob_dist = cl.prob_classify("This one's a doozy.")
prob_dist.max()
round(prob_dist.prob("pos"), 2)
round(prob_dist.prob("neg"), 2)