# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:01:33 2020

@author: Khamar Uz Zama

This program reads the pre-processed data, extracts various
features from it and saves them for future use.
"""

import preProcessing as pp
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import datefinder
import nltk
import pandas as pd
import numpy as np
import functools
import operator
import textstat
import os

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date, datetime, time, timedelta
from lexicalrichness import LexicalRichness


nltk.download('averaged_perceptron_tagger')

ppFile = "processedHTMLnoLemma.pickle"
sentsFile = "sentiments.pickle"
datesFile = "dates.pickle"
posFile = 'posNouns.pickle'
FRSFile = 'FRSscores.pickle'
lexRichFile = 'lexRich.pickle'

# noOfFilesToLoad = -1 to read all files
noOfFilesToLoad = -1
savepreProcessingData = False
loadpreProcessingData = True
data = pp.readIndexes()

# define the configuration for preprocessing
preProcessingConfig = {
        "lower":True,
        "symbols":True,
        "lemmatize":False,
        "stem":False,
        "stopWords":True
        }

try:
    """
    Load the data, if any exception is encountered, the given number of
    documents are processed again and saved.
    """
    if(loadpreProcessingData):
        with open(ppFile, 'rb') as f:
            processedData = pickle.load(f)
            labels = processedData[-1]
            
            # Remove labels from data
            processedData.pop()

            if(noOfFilesToLoad != -1):
                processedData = processedData[:noOfFilesToLoad]
                labels = labels[:noOfFilesToLoad]
                
    else:
        raise Exception
except:
    print('preprocessing HTML files')
    processedData = pp.processHTMLFiles(noOfFilesToLoad, preProcessingConfig)
    labels = processedData[-1]

    if(noOfFilesToLoad == -1):
        noOfFilesToLoad = labels.shape[0]
    
    if(savepreProcessingData):
        with open(ppFile, 'wb') as f:
            pickle.dump(processedData,f)
            
    processedData.pop()



    

def extractsentiments():
    """
    Extract polarity scores for the given data.
    The polartiy scores extracted are the overall sentiments 
    (Positive ,Negative ,Neutral) for the whole book (HTML file)
    """
    sid = SentimentIntensityAnalyzer()
    
    sentiments = []
    
    for i,doc in enumerate(processedData):
        print(i)
        sentences = ",".join(doc)
        sentScores = sid.polarity_scores(sentences)
         
        sentiments.append(sentScores)

    with open(sentsFile, 'wb') as f:
        pickle.dump(sentiments,f)
        
def extractDates():
    """
    Extract dates from the processed data and pickle them.
    """    
    datesList = []
    for i,doc in enumerate(processedData):
        dates = []
        x = ",".join(doc)
    
        matches = datefinder.find_dates(x)
        for match in matches:
            dates.append(match)
            
        mylist = list(dict.fromkeys(dates))
        datesList.append(mylist)

    with open(datesFile, 'wb') as f:
        pickle.dump(datesList,f)

def extractPOS(doc):
    """
    Extract the count of nouns from the given document
    """
    
    is_noun = lambda pos: pos[:2] == 'NN'

    count = 0
    for sentence in doc:        
        tokenized = nltk.word_tokenize(sentence)
        for (word, pos) in nltk.pos_tag(tokenized):
            if is_noun(pos):
                count +=1
    
    return count

def extractPOSAllHTMLFiles():
    """
    Extract the count of nouns from all the documents using the helper function
    """
    
    nouns = []
    for i,doc in enumerate(processedData):
        print(i)
        temp = extractPOS(doc)
        nouns.append(temp)
    
    with open(posFile, 'wb') as f:
        pickle.dump(nouns,f)
        
    return nouns

def avg_datetime(series, ind):
    res = list(filter(None, series))
    zzzz = []
    for i,x in enumerate(np.logical_not(pd.isnull(res))):
        if(x):
            zzzz.append(res[i])
    asd = pd.Series(zzzz)
    dt_min = asd.min()
    
    deltas = [x-dt_min for x in asd]
    pd.to_datetime(deltas)
    x = functools.reduce(operator.add, deltas)
    x = sum((c for c in deltas), timedelta())
    try:
        return dt_min + functools.reduce(operator.add, deltas) / len(deltas)
    except:
        print()
        return dt_min + functools.reduce(operator.add, deltas) / len(deltas)



def getAverageDates():
    with open(datesFile, 'rb') as f:
        dates = pickle.load(f)    
    
    x = pd.DataFrame(dates)
    
#    y = x.loc[:4]
#    z = pd.DataFrame()
#    for (columnName, columnData) in y.iteritems():
#        asdates = columnData.values
#        z.columnName = asdates
    avg = []
    for ind in x.index:
         avg.append(avg_datetime(x.iloc[ind,:], ind))
         
    return avg
 
def extractFRSAllHTMLFiles():
    """
    Extract Fleisch Reading Scores (FRS).
    The documents and indexes are read again as the FRS is given for the 
    original text and not the processed text
    """
    
    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    HTMLFilesPath = 'Gutenberg_19th_century_English_Fiction'
    FRSScores = []
    badIndexes = []
    dataPath = os.path.join(os.getcwd(),Path1,Path2, HTMLFilesPath)
    data = pp.readIndexes()
    
    for i in range(len(data)):
        print(i)
        htmlFilePath = os.path.join(dataPath,data['book_id'][i])[:-5] + '-content.html'
        corpus = pp.readHTMLFile(htmlFilePath)
        if corpus:
            score = textstat.flesch_reading_ease(corpus)

            FRSScores.append(score)
        else:
            badIndexes.append(i)
            
    with open(FRSFile, 'wb') as f:
        pickle.dump(FRSScores,f)
        
        
def extractLexicalRichness():
    """
    Extract lexical richness of the text documents.
    The original text is used here too.
    """
    
    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    HTMLFilesPath = 'Gutenberg_19th_century_English_Fiction'
    lexScores = []
    badIndexes = []
    dataPath = os.path.join(os.getcwd(),Path1,Path2, HTMLFilesPath)
    data = pp.readIndexes()
    
    for i in range(len(data)):
        print(i)
        htmlFilePath = os.path.join(dataPath,data['book_id'][i])[:-5] + '-content.html'
        corpus = pp.readHTMLFile(htmlFilePath)
        if corpus:
            lex = LexicalRichness(corpus)
            del lex.wordlist
            del lex.text
            
            lexScores.append(lex)
        else:
            badIndexes.append(i)
            
    with open(lexRichFile, 'wb') as f:
        pickle.dump(lexScores,f)


def loadData():
    """
    Loads all the pickle files and returns them
    """
    
    with open(sentsFile, 'rb') as f:
        sents = pickle.load(f)
        
    with open(datesFile, 'rb') as f:
        dates = pickle.load(f)
    
    with open(posFile, 'rb') as f:
        nns = pickle.load(f)
        
    with open(FRSFile, 'rb') as f:
        FRSScores = pickle.load(f)
        
    with open(lexRichFile, 'rb') as f:
        lexRich = pickle.load(f)
                
    return sents, dates, nns, FRSScores, lexRich
        
def buildFeatureVector():
    """
    Builds the feature vector (including labels) using all 
    pickle files and saves it as a CSV file
    """
    sents, dates, nns, FRSScores, lexRich = loadData()
    featureMatrix = []
    featureNames = ['neg', 'neu', 'pos', 'nns', 'FRS', 'Dugast', 'Herdan', 'Maas', 'Summer', 'cttr', 'rttr', 'label']
    
    for i,y in enumerate(labels):
        featureVector = []
        featureVector.append(sents[i]['neg'])
        featureVector.append(sents[i]['neu'])
        featureVector.append(sents[i]['pos'])
        featureVector.append(nns[i])
        featureVector.append(FRSScores[i])
        featureVector.append(lexRich[i].Dugast)
        featureVector.append(lexRich[i].Herdan)
        featureVector.append(lexRich[i].Maas)
        featureVector.append(lexRich[i].Summer)
        featureVector.append(lexRich[i].cttr)
        featureVector.append(lexRich[i].rttr)
        featureVector.append(y)
        featureMatrix.append(featureVector)
        print(i)
    
    data = pd.DataFrame(featureMatrix,columns=featureNames)
    
    return data

if __name__ == "__main__":
    
#    plotGenres()   
#    extractsentiments()
#    extractDates()
#    asdasda = getAverageDates()    
#    extractPOSAllHTMLFiles()
#    extractFRSAllHTMLFiles()
#    extractLexicalRichness()
    
#    data = buildFeatureVector()
#    data.to_csv('data.csv')