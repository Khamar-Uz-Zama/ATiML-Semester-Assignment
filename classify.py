# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:01:33 2020

@author: Khamar Uz Zama
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
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from datetime import date, datetime, time, timedelta



nltk.download('averaged_perceptron_tagger')

ppFile = "processedHTMLnoLemma.pickle"
sentsFile = "sentiments.pickle"
datesFile = "dates.pickle"
posFile = 'posNouns.pickle'


# noOfFilesToLoad = -1 for all files
noOfFilesToLoad = -1
savepreProcessingData = False
loadpreProcessingData = True
data = pp.readIndexes()
preProcessingConfig = {
        "lower":True,
        "symbols":True,
        "lemmatize":False,
        "stem":False,
        "stopWords":True
        }

try:
    if(loadpreProcessingData):
        with open(ppFile, 'rb') as f:
            processedData = pickle.load(f)
            labels = processedData[-1]
    
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

def plotGenres():
    targetCounts = labels.value_counts()
    ax = sns.barplot(x=targetCounts.index, y=targetCounts.values, palette="Blues_d")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X, y)
    return svm

def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 
    """
    concDocs = []
    separator = ','
    
    for i,doc in enumerate(processedData):
        print(i)
        concDocs.append(separator.join(doc))
        
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(concDocs)
    
    return X, vectorizer

def svmClassifier():

    vectorizedData, vectorizer = create_tfidf_training_data(processedData)

    X_train, X_test, y_train, y_test = train_test_split(
        vectorizedData, labels, test_size=0.2, random_state=42
    )

    svm = train_svm(X_train, y_train)
    
    preds = svm.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, preds)
    cf = metrics.confusion_matrix(y_test, preds)

    plt.figure(figsize = (10,7))
    sns.heatmap(cf, annot=True)   
    print("Accuracy achieved using svm = {}".format(accuracy))

def extractsentiments():
    sid = SentimentIntensityAnalyzer()
    
    sentiments = []
    
    for i,doc in enumerate(processedData):
        print(i)
        x = ",".join(doc)
        ss = sid.polarity_scores(x)
#        for k in sorted(ss):
#         print('{0}: {1}, '.format(k, ss[k]), end='')
#         print()
         
        sentiments.append(ss)
        

    with open(sentsFile, 'wb') as f:
        pickle.dump(sentiments,f)
        
def extractDates():
    
    zz = []
    for i,doc in enumerate(processedData):
        dates = []
        x = ",".join(doc)
    
        matches = datefinder.find_dates(x)
        for match in matches:
            dates.append(match)
            
        mylist = list(dict.fromkeys(dates))
        zz.append(mylist)

    with open(datesFile, 'wb') as f:
        pickle.dump(zz,f)

def loadData():
    
    with open(sentsFile, 'rb') as f:
        sents = pickle.load(f)
        
    with open(datesFile, 'rb') as f:
        dates = pickle.load(f)
    
    with open(posFile, 'rb') as f:
        nns = pickle.load(f)
        
    return sents, dates, nns

def extractPOS(doc):
    is_noun = lambda pos: pos[:2] == 'NN'
#    for sentence in doc:        
#        tokenized = nltk.word_tokenize(sentence)
#        snouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
#        dnouns.append(snouns)
    x = 0
    for sentence in doc:        
        tokenized = nltk.word_tokenize(sentence)
        for (word, pos) in nltk.pos_tag(tokenized):
            if is_noun(pos):
                x +=1
    return x

def extractPOSAllHTMLFiles():
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
    
    y = x.loc[:4]
#    z = pd.DataFrame()
#    for (columnName, columnData) in y.iteritems():
#        asdates = columnData.values
#        z.columnName = asdates
    avg = []
    for ind in x.index:
         avg.append(avg_datetime(x.iloc[ind,:], ind))
         
    return avg

if __name__ == "__main__":
    
#    svmClassifier()
#    plotGenres()   
#    extractsentiments()
#    extractDates()    
#    extractPOSAllHTMLFiles()
    asdasda = getAverageDates()
    loadData()
    
