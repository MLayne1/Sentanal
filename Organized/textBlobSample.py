from textblob.classifiers import NaiveBayesClassifier

# train =[
#     ('I love this sandwich.', 'pos'),
#     ('this is an amazing place!', 'pos'),
#     ('I feel very good about these beers.', 'pos'),
#     ('this is my best work.', 'pos'),
#     ("what an awesome view", 'pos'),
#     ('I do not like this restaurant', 'neg'),
#     ('I am tired of this stuff.', 'neg'),
#     ("I can't deal with this", 'neg'),
#     ('he is my sworn enemy!', 'neg'),
#     ('my boss is horrible.', 'neg')
# ]

# cl = NaiveBayesClassifier(train)

print("Running!")

with open('train.json', encoding='utf-8', mode='r') as train:
    cl = NaiveBayesClassifier(train, format="json")


print(cl.show_informative_features(5))

with open('test.json', encoding='utf-8', mode='r') as test:
    print("accuracy: " + str(cl.accuracy(test, format="json")))


# toClassify = input("classify: ")
# print(cl.classify(toClassify))



print("Done!!")

# prob_dist = cl.prob_classify("This one's a doozy.")

# v = prob_dist.max()

# print(v)

# v = round(prob_dist.prob("pos"), 2)
# print(v)

# v = round(prob_dist.prob("neg"), 2)
# print(v)
