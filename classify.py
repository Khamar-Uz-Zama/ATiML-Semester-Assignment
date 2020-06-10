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
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('averaged_perceptron_tagger')

ppFile = "processedHTMLnoLemma.pickle"
sentsFile = "sentiments.pickle"
datesFile = "dates.pickle"

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

def sentimentAnalysis():
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
        print(i)
        for match in matches:
            dates.append(match)
            
        mylist = list(dict.fromkeys(dates))
   
        zz.append(mylist)
        

    with open(datesFile, 'wb') as f:
        pickle.dump(zz,f)

def loadData():
    
    with open(sentsFile, 'rb') as f:
        sents = pickle.load(f)
        
    print("")

    with open(datesFile, 'rb') as f:
        dates = pickle.load(f)        
    
    return sents, dates

def extractPOS(text):

    lines = 'lines is some string of words'
    is_noun = lambda pos: pos[:2] == 'NN'
    nouns = [word for (word, pos) in nltk.pos_tag(text) if is_noun(pos)]
    
    return nouns

def extractPOSAllHTMLFiles():
    nouns = []
    for i,doc in enumerate(processedData):
        temp = extractPOS(doc)
        nouns.append(temp)
    return nouns

if __name__ == "__main__":
    
#    svmClassifier()
#    plotGenres()   
#    extractsentiments()
#    extractDates()    

#   loadData()

    extractPOS(processedData[0])
        