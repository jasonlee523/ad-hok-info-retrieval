import sys
import math
import re
import operator
from stop_list import closed_class_stop_words as stop_list
qTFIDF = []     #Query TFIDF list
aTF = []        #Abstract TF list
aIDF = {}       #Abstract IDF list

with open('cran.qry') as instream:
    queries = []
    qTF = []
    qIDF = {}

    text = instream.read()
    lines = re.split('\.I\s\d{3}\n\.W', text)       #Split queries into a list
    for line in lines:                              #Refine queries
        line = line.strip('.\n')
        line = line.replace('\n', ' ')
        words = line.split(' ')
        for word in words[:]:
            if word in stop_list:                   #Remove stop list words
                words.remove(word)
            if re.compile('[\W\d]').search(word):   #Remove punctuation and digits
                words[words.index(word)] = re.sub('[\W\d]', '', word)
        words.pop()
        queries.append(words)

    qWords = {}
    for query in queries:       #Build a baseline vector of all words in the queries
        for word in query:
            if word not in qWords:
                qWords[word] = 0;

    numQueriesContainingT = qWords.copy()
    for query in queries:       #Build query TF and find number of query docs containing T
        qVector = qWords.copy()
        for word in query:
            qVector[word] += 1;
            numQueriesContainingT[word] += 1;
        qTF.append(qVector);

    for term in numQueriesContainingT:     #Build query IDF
        qIDF[term] = math.log(225/numQueriesContainingT[term])

    for query in qTF:            #Build query TFIDF
        vector = {}
        for word in query:
            vector[word] = query[word] * qIDF[word] / math.log(len(query))  #Modify TF based on query length
        qTFIDF.append(vector)


with open('cran.all.1400') as instream:
    abstracts = []
    text = instream.read()
    lines = re.split('\.I[\s\S]*?\.W', text)    #Split abstracts into a list
    for line in lines:                          #Refine abstracts
        line = line.strip('\n')
        words = line.split(' ')
        for word in words:                      #Remove punctuation and numbers
            if re.compile('[\W\d]').search(word):
                words[words.index(word)] = re.sub('[\W\d]', '', word)
        for word in words[:]:                   #Remove empty strings
            if not word:
                words.remove(word)
        for word in words[:]:                   #Remove stop words
            if word in stop_list:
                words.remove(word)
        abstracts.append(words)

    aWords = {}
    for abstract in abstracts:       #Build a baseline vector of all words in the abstracts
        for word in abstract:
            if word not in aWords:
                aWords[word] = 0;

    numDocsContainingT = aWords.copy()
    for abstract in abstracts:       #Build abstract TF list and find number of abtracts containing T
        vector = aWords.copy()
        for word in abstract:
            vector[word] += 1;
            numDocsContainingT[word] += 1;
        aTF.append(vector);

    for term in numDocsContainingT:     #Build abstract IDF
        aIDF[term] = math.log(1400/numDocsContainingT[term])


output = []
for vector in qTFIDF:
    scores = {}
    for TF in aTF:
        aTFIDF = {}
        for word in vector:               #Build array of terms that match query terms containing TF-IDF scores
            if word in TF:
                aTFIDF[word] = TF[word] * aIDF[word] / math.log(len(TF))    #Modify score based on document length
            else:
                aTFIDF[word] = 0

        sumOfProducts = 0
        sumOfQueries = 0
        sumOfAbstracts = 0
        for word in vector:                 #Build summations for cosine similarity
            sumOfProducts += vector[word] * aTFIDF[word]
            sumOfQueries += math.pow(vector[word], 2)
            sumOfAbstracts += math.pow(aTFIDF[word], 2)
        QAproduct = sumOfQueries * sumOfAbstracts
        if sumOfProducts > 0:                #add query-document scores using cos similarity
            scores[aTF.index(TF)] = sumOfProducts / math.sqrt(QAproduct)
        else:
            scores[aTF.index(TF)] = 0
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)    #sort scores
    output.append(sorted_scores)

f = open('output.txt', 'w')
for query in output:
    for doc, score in query:
        f.write(str(output.index(query)) + ' ' + str(doc) + ' ' + str(score) + '\n')
f.close()
