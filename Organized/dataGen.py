import os
import re

pathTrainFake = '.\\FakeBuzzfeed'
pathTrainReal = '.\\RealBuzzfeed'
pathTestFake = '.\\FakeRandom'
pathTestReal = '.\\RealRandom'

jsonTrain = "train.json"
jsonTest = "test.json"

if os.path.exists(jsonTrain):
    os.remove(jsonTrain)

if os.path.exists(jsonTest):
    os.remove(jsonTest)

def addToJson(jsonName, pathToFolder, label, last):
    with open(jsonName, encoding="utf8", mode='a') as the_file:
        count = 1
        nFiles = len(os.listdir(pathToFolder))
        print("adding: " + str(nFiles) + "files")
        if not last:
            the_file.write('[\n') 
        for doc in os.listdir(pathToFolder):
            if doc.endswith(".txt"):
                print (str(count) + ":" +"adding: " + doc + " to " + jsonName)
                toRead = open(pathToFolder + "\\" + doc, 'r')
                the_file.write("\t{ \"text\": \"")
                text = toRead.read()
                text = re.sub('[^a-zA-Z ]', ' ', text)
                text = re.sub(' +', ' ', text)
                the_file.write(text)
                if last and (count == nFiles):
                    print ("cool")
                    the_file.write("\", \"label\": \"" + label + "\"}\n")
                else:
                    the_file.write("\", \"label\": \"" + label + "\"},\n")
                toRead.close()
                count += 1
                continue
            else:
                continue
        if last:
            the_file.write(']')   
    
addToJson(jsonTrain, pathTrainFake, "neg", False)
addToJson(jsonTrain, pathTrainReal, "pos", True)
addToJson(jsonTest, pathTestFake, "neg", False)
addToJson(jsonTest, pathTestReal, "pos", True)
