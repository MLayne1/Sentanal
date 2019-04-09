import pandas as pd 
import json
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

"""
Parse TSV into desired JSON
"""
# source path
SRC_REAL = 'articles_real.tsv'
SRC_FAKE = 'articles_fake.tsv'

# output path
OUT_REAL = 'jRealRaw.json'
OUT_FAKE = 'jFakeRaw.json'

# labels
LBL_REAL = 'pos'
LBL_FAKE = 'neg'

def clean(toClean):
    """ This function cleans the text before adding it to the JSON.
    It operates by using regex to remove non-alphabetical charecters
    and extra spcaces, then uses NLTK to remove stop words

    Arguments:
        toClean {str} -- A string of text to be cleaned
    Returns:
        {str} -- A cleaned string of text
    """

    stemmer = EnglishStemmer()
    stopWords = set(stopwords.words('english'))
    clean = ""

    # Force lowercase
    toClean = toClean.lower()

    # remove non alphabetical chars and remove extra spaces
    toClean = re.sub('[^a-z,.!?]', ' ', toClean) # use [^a-zA-Z] if not forced lowercase
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

def generateDataFrame(source, label):
    """Generates pandas DataFrame

    Arguments:
        source {str} -- path to the source TSV
        label {str} -- the label for the data
    Return
        {DataFrame} -- Pandas DataFrame with text and label
    """
    dataFrame = pd.read_csv(source, sep='\t', encoding='utf-8', names=["URL", "text", "label"], index_col=False)
    # Remove URL
    dataFrame = dataFrame.drop(columns="URL")
    # Add label
    dataFrame["label"] = label
    # remove empty cells
    dataFrame.replace('', np.nan, inplace=True)
    dataFrame.dropna(inplace=True)

    return dataFrame

def cleanJson(path):
    """ Clean JSON file destructively (overwrite file)

    Arguments:
        path {str} -- path to the JSON file to be cleaned
    """
    with open(path) as jsonFile:
        jsonObj = json.load(jsonFile)
        i = 0
        while i < len(jsonObj):
            # clean text
            jsonObj[i]['text'] = clean(jsonObj[i]['text'])
            i+=1
    with open(path, 'w') as jsonFile:
        json.dump(jsonObj, jsonFile, indent=4)

# generate pandas dataFrames
jReal = generateDataFrame(SRC_REAL, LBL_REAL)
jFake = generateDataFrame(SRC_FAKE, LBL_FAKE)

# create intermediate JSON file
out1 = jReal.to_json(OUT_REAL, orient='records')
out2 = jFake.to_json(OUT_FAKE, orient='records')

# clean output
# cleanJson(OUT_REAL)
# cleanJson(OUT_FAKE)