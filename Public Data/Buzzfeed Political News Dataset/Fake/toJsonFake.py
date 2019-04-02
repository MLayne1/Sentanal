import os

path = 'C:/Users/ASUS/Desktop/Public Data/Buzzfeed Political News Dataset/Fake'

if os.path.exists("fake.json"):
    os.remove("fake.json")


with open('fake.json', encoding="utf8", mode='a') as the_file:
        the_file.write('[\n')
        
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                print (filename)
                toRead = open(filename, 'r')
                the_file.write("\t{ \"text\": \"")
                
                text = toRead.read()
                text = text.replace('\n', ' ')
                text = text.replace('\r', ' ')
                text = text.replace('\"', '\'')
                text = text.replace("\\", ' ')
                text = text.replace("/", ' ')

                the_file.write(text)

                the_file.write("\", \"label\": \"Fake\"},\n")
                toRead.close()
                continue
            else:
                continue
        the_file.write('\n]')



