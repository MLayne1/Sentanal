import nltk
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()

def train():
	trainingData = open("trainingData.txt", "r")

	for line in trainingData:
		values = line.split(", ")
		for value in values:
			value.strip("\n")
			print(value)

def sentimentCheck(word):
    print("Token: " + word)
    print("Stemmed: " + stemmer.stem(word) + "\n")

def test():
    sentence = str(input())
    tokens = nltk.word_tokenize(sentence)

    print("\nList of Tokens: ")
    print(str(tokens) + "\n")

    for token in tokens:
       sentimentCheck(token)

test()

print("\n\n\n")

train()