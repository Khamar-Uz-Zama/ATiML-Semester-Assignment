# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:01:33 2020

@author: Khamar Uz Zama
"""
import preProcessing as pp
import pickle
import seaborn as sns
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

filename = "processedHTML.pickle"
noOfFilesToLoad = 100
pickleData = False

data = pp.readIndexes()

try:   
    with open(filename, 'rb') as f:
        processedData = pickle.load(f)
        if(noOfFilesToLoad != -1):
            processedData = processedData[:noOfFilesToLoad]
            labels = data['guten_genre'][:noOfFilesToLoad]
        
except:
    print('No pickle found, preprocessing {} HTML files', noOfFilesToLoad)
    processedData = pp.processAllHTMLFiles(data,noOfFilesToLoad)
    if(pickleData):
        with open(filename, 'wb') as f:
            pickle.dump(processedData,f)

z = data['guten_genre'].value_counts()
ax = sns.barplot(x=z.index, y=z.values, 
                 palette="Blues_d")



def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm


def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 
    """
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(docs)
    
    return X

if __name__ == "__main__":
    # Vectorise and TF-IDF transform the corpus 
    vectorizedData = create_tfidf_training_data(processedData)

    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(
        vectorizedData, labels, test_size=0.2, random_state=42
    )

    # Create and train the Support Vector Machine
    svm = train_svm(X_train, y_train)
    
