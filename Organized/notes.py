import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

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

    # remove non alphabetical chars and remove extra spaces
    toClean = re.sub('[^a-z]', ' ', toClean) # use [^a-zA-Z] if not forced lowercase
    # Replaces sequential spaces with single space
    toClean = re.sub(' +', ' ', toClean)

    # tokenize words for stemming and stop word removal
    tokens = word_tokenize(toClean)

    # for each token, stem token and then check if it is stop word.
    # adds token to the clean text if it is not a stop word.
    for token in tokens:
        # token = stemmer.stem(token)
        if token not in stopWords:
            clean = clean + (str(token) + " ")

    return clean


    
                text = clean(text)