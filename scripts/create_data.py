import requests
import json
from random import randint

NUMBER_OF_FALSE_PAIRS = 1000000
solr_server = 'http://localhost:8983/solr/'
col_name = 'your-collection-name/'
number = '3462432' #Number of documents that your collection has
url_query = solr_server + col_name + 'select?indent=on&q=lang:de&wt=json&rows=' + number + '&fl=message%20author%20recipient%20responseTo%20questionTo%20'


DOWNLOAD_DATA_FROM_SOLR = True #If already downloaded set to False

if (DOWNLOAD_DATA_FROM_SOLR):
    print("Requesting information...")
    r = requests.get(url_query).json()['response']['docs']
    print("Information retrieved")
    with open('my_data.json', 'w') as outfile:
        json.dump(r, outfile)

def addToSet(file, message, context, label):
    to_write = '"' + context + '",' + '"' +  message + '",' + str(label) + '\n'
    file.write(to_write)

with open('my_data.json', 'r') as outfile:
    data_points = json.load(outfile)

    print("NUMBER OF DOCUMENTS: " + str(len(data_points)))
    buffer_mess = ''


    with open('../data/train.csv', 'w+') as train:
        # Here we add the header of the train.csv file
        train.write('Context,Utterance,Label' + '\n')

        # Here we add the correct Context-Utterance pairs of our data to the train.csv file (with label 1)
        for n in range(len(data_points)-1):
            if n%100000==0:
                print(str(n) + " candidates done!")

            data = data_points[n]
            nextData = data_points[n+1]

            if 'responseTo' not in  data.keys() or 'message' not in data.keys():
                continue

            message = data['message']
            responseTo = data['responseTo']
            if 'responseTo' not in  nextData.keys():
                nextResponseTo = ''
            else:
                nextResponseTo = nextData['responseTo']
            if responseTo == nextResponseTo:
                if len(buffer_mess)>0 and buffer_mess[-1] != '.':
                    buffer_mess += '. ' + message
                elif buffer_mess=='':
                    buffer_mess = message
                else:
                    buffer_mess += ' ' + message
            else:
                if len(buffer_mess)>0 and buffer_mess[-1] != '.':
                    buffer_mess += '. ' + message
                elif buffer_mess=='':
                    buffer_mess = message
                else:
                    buffer_mess += ' ' + message
                context = ' '.join(responseTo)
                addToSet(train, buffer_mess.replace('"', "'"), context.replace('"', "'"), 1)
                buffer_mess = ''

        length = len(data_points)

        # Here we add the false Context-Utterance pairs of our data to the train.csv file (with label 0)
        for n in range(NUMBER_OF_FALSE_PAIRS):
            if n%100000==0:
                print(str(n) + " false candidates done!")
            
            rand1 = randint(0, length-1)
            rand2 = randint(0, length-1)
            data = data_points[rand1]
            nextData = data_points[rand2]

            if 'responseTo' not in  data.keys() or 'message' not in nextData.keys():
                continue

            message = nextData['message']
            responseTo = data['responseTo']

            context = ' '.join(responseTo)
            addToSet(train, message.replace('"', "'"), context.replace('"', "'"), 0)

