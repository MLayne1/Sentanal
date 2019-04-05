""" Quick script meant to merge train and test into fake and real """

import json

SRC_TRAIN = '.\\train.json'
SRC_TEST = '.\\test.json'

OUT_FAKE = '.\\hFakeRaw.json'
OUT_REAL = '.\\hRealRaw.json'


def makeJsons(srcTest, srcTrain):

    oFake = []
    oReal = []

    with open(srcTest) as jFile:
        tj = json.load(jFile)
        for article in tj:
            if article['label'] == 'pos':
                oReal.append(article)
            else:
                oFake.append(article)

    with open(srcTrain) as jFile:
        fj = json.load(jFile)
        for article in fj:
            if article['label'] == 'pos':
                oReal.append(article)
            else:
                oFake.append(article)

    with open(OUT_FAKE, 'w') as jsonFile:
        json.dump(oFake, jsonFile, indent=4)

    with open(OUT_REAL, 'w') as jsonFile:
        json.dump(oReal, jsonFile, indent=4)

makeJsons(SRC_TEST, SRC_TRAIN)