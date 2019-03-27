import nltk

def train():
	pass

def sentimentCheck(word):
    print(word)


def test():
    sentence = str(input())
    tokens = nltk.word_tokenize(sentence)

    for token in tokens:
        sentimentCheck(token)

test()