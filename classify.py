# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:01:33 2020

@author: Khamar Uz Zama
"""

import preProcessing as pp
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer

filename = "processedHTMLnoLemma.pickle"
# noOfFilesToLoad = -1 for all files
noOfFilesToLoad = -1
saveData = True
loadData = True
data = pp.readIndexes()
preProcessingConfig = {
        "lower":True,
        "symbols":True,
        "lemmatize":False,
        "stem":False,
        "stopWords":True
        }

try:
    if(loadData):
        with open(filename, 'rb') as f:
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
    
    if(saveData):
        with open(filename, 'wb') as f:
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

if __name__ == "__main__":
    
#    svmClassifier()
#   plotGenres()   
    
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
        

    with open("sentiments.pickle", 'wb') as f:
        pickle.dump(sentiments,f)
        
    with open("sentiments.pickle", 'rb') as f:
        sents = pickle.load(f)
    print("")