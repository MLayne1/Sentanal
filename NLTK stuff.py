import nltk

def train():
	pass

def sentimentCheck(word):
    print("Token: " + word)


def test():
    sentence = str(input())
    tokens = nltk.word_tokenize(sentence)

    print("\nList of Tokens: ")
    print(tokens)
    print()

    for token in tokens:
        sentimentCheck(token)

test()