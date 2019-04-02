import os
import re


"""
Script that generates a training and testing JSON file given some directories where each data point is a single .txt file
"""

# paths to labeled data 
pathTrainFake = '.\\FakeBuzzfeed'
pathTrainReal = '.\\RealBuzzfeed'
pathTestFake = '.\\FakeRandom'
pathTestReal = '.\\RealRandom'

# names for output files
jsonTrain = "train.json"
jsonTest = "test.json"

# remove files if they already exist
if os.path.exists(jsonTrain):
    os.remove(jsonTrain)
if os.path.exists(jsonTest):
    os.remove(jsonTest)



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
        print("adding: " + str(nFiles) + " files")
        if not last:
            the_file.write('[\n') 
        for doc in os.listdir(pathToFolder):
            if doc.endswith(".txt"):
                print (str(count) + "/" + str(nFiles) +": " + doc + " to " + jsonName)
                toRead = open(pathToFolder + "\\" + doc, 'r')
                the_file.write("\t{ \"text\": \"")
                text = toRead.read()
                text = re.sub('[^a-zA-Z ]', ' ', text)
                text = re.sub(' +', ' ', text)
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
