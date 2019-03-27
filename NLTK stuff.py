import nltk

sentence = str(input())
tokens = nltk.word_tokenize(sentence)


def train():


def sentimentCheck(word):
	print(word)


def test():
	sentence = str(input())
	tokens = nltk.word_tokenize(sentence)

	for token in tokens:
		sentimentCheck(token)
