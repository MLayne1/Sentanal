from textblob.classifiers import NaiveBayesClassifier

train =[
    ('I love this sandwich.', 'pos'),
    ('this is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('he is my sworn enemy!', 'neg'),
    ('my boss is horrible.', 'neg')
]

# with open('train.json', 'r') as fp:
#     cl = NaiveBayesClassifier(fp, format="json")

cl = NaiveBayesClassifier(train)

cl.classify("This is an amazing library!")

prob_dist = cl.prob_classify("This one's a doozy.")
prob_dist.max()
round(prob_dist.prob("pos"), 2)
round(prob_dist.prob("neg"), 2)