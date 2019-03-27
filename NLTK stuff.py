import nltk
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()

def train():
	pass

def sentimentCheck(word):
    print("Token: " + word)
    print("Stemmed: " + stemmer.stem(word) + "\n")


def test():
    sentence = str(input())
    tokens = nltk.word_tokenize(sentence)


    print("\nList of Tokens: ")
    print(tokens)
    print()

    for token in tokens:
       sentimentCheck(token)

test()