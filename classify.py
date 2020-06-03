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


filename = "processedHTML.pickle"
# noOfFilesToLoad = -1 for all files
noOfFilesToLoad = -1
pickleData = True

data = pp.readIndexes()


try:
    with open(filename, 'rb') as f:
        processedData = pickle.load(f)
        if(noOfFilesToLoad != -1):
            processedData = processedData[:noOfFilesToLoad]
        labels = processedData[-1]

except:
    print('No pickle found, preprocessing HTML files')
    processedData = pp.processAllHTMLFiles(noOfFilesToLoad)
    labels = processedData[-1]

    if(noOfFilesToLoad == -1):
        noOfFilesToLoad = labels.shape[0]
    
    if(pickleData):
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

    # Vectorise and TF-IDF transform the corpus 
    vectorizedData, vectorizer = create_tfidf_training_data(processedData)

    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(
        vectorizedData, labels, test_size=0.2, random_state=42
    )

    # Create and train the Support Vector Machine
    svm = train_svm(X_train, y_train)
    
    preds = svm.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, preds)
    cf = metrics.confusion_matrix(y_test, preds)

    plt.figure(figsize = (10,7))
    sns.heatmap(cf, annot=True)    

if __name__ == "__main__":
    
    svmClassifier()
    