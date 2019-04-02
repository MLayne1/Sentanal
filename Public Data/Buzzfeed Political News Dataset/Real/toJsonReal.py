import os
import re

path = '.'

if os.path.exists("real.json"):
    os.remove("real.json")


with open('real.json', encoding="utf8", mode='a') as the_file:
        the_file.write('[\n')
        
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                print (filename)
                toRead = open(filename, 'r')
                the_file.write("\t{ \"text\": \"")
                
                text = toRead.read()
                text = re.sub('[^a-zA-Z ]', ' ', text)
                text = re.sub(' +', ' ', text)

                the_file.write(text)

                the_file.write("\", \"label\": \"Real\"},\n")
                toRead.close()
                continue
            else:
                continue
        the_file.write('\n]')



