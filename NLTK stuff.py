import nltk
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()

def train():
	trainingData = open("trainingData.txt", "r")

	for line in trainingData:
		values = line.split(", ")
		values[1] = values[1].strip("\n")
		print(values)

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

# Calls methods for testing them
#test()

train()