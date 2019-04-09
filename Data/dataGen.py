import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

"""
Script that generates a training and testing JSON file given some
directories where each data point is a single .txt file
"""

# paths to labeled data 
pathTrainFake = '.\\FakeBuzzfeed'
pathTrainReal = '.\\RealBuzzfeed'
pathTestFake = '.\\FakeRandom'
pathTestReal = '.\\RealRandom'

# names for output files
jsonTrain = "jsonFiles\\train.json"
jsonTest = "jsonFiles\\test.json"

# remove files if they already exist
if os.path.exists(jsonTrain):
    os.remove(jsonTrain)
if os.path.exists(jsonTest):
    os.remove(jsonTest)

def clean(toClean):
    """
    This function cleans the text before adding it to the JSON.
    It operates by using regex to remove non-alphabetical charecters
    and extra spcaces, then uses NLTK to remove stop words
    """

    stemmer = EnglishStemmer()
    stopWords = set(stopwords.words('english'))
    clean = ""

    # Force lowercase
    toClean = toClean.lower()

    # remove non alphabetical chars and remove extra spaces esxcept ',' '.' and ' '
    toClean = re.sub('[^a-z., ]', ' ', toClean)
    # Replaces sequential spaces with single space
    toClean = re.sub(' +', ' ', toClean)

    # tokenize words for stemming and stop word removal
    tokens = word_tokenize(toClean)

    # for each token, stem token and then check if it is stop word.
    # adds token to the clean text if it is not a stop word.
    for token in tokens:
        token = stemmer.stem(token)
        if token not in stopWords:
            clean = clean + (str(token) + " ")

    return clean



def addToJson(jsonName, pathToFolder, label, last):
    """
    adds multiple text files to a single json file,
        jsonName: name of output file
        pathToFoler: path to the folder containing the labeled data
        label: the label of the data
        last: boolean to indicate if this is the last folder being added (current implementation assumes you're only adding two folder to the json)
    """
    with open(jsonName, encoding="utf8", mode='a') as the_file:
        count = 1
        nFiles = len(os.listdir(pathToFolder))
        print("\nAdding: " + str(nFiles) + " files")
        if not last:
            the_file.write('[\n') 
        for doc in os.listdir(pathToFolder):
            if doc.endswith(".txt"):
                print (str(count) + "/" + str(nFiles) +": " + doc + " to " + jsonName)
                toRead = open(pathToFolder + "\\" + doc, 'r')
                the_file.write("\t{ \"text\": \"")
                text = toRead.read()
                text = clean(text)
                the_file.write(text)
                if last and (count == nFiles):
                    the_file.write("\", \"label\": \"" + label + "\"}\n")
                else:
                    the_file.write("\", \"label\": \"" + label + "\"},\n")
                toRead.close()
                count += 1
        if last:
            the_file.write(']')
    
addToJson(jsonTrain, pathTrainFake, "neg", False)
addToJson(jsonTrain, pathTrainReal, "pos", True)
addToJson(jsonTest, pathTestFake, "neg", False)
addToJson(jsonTest, pathTestReal, "pos", True)
