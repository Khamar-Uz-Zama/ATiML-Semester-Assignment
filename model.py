# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:36:25 2020

@author: Khamar Uz Zama
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras import layers

data = pd.read_csv('data.csv', index_col=False)
data = data.drop(data.columns[0],1)
X = data.iloc[:,:-1]
labels = data.iloc[:,-1]
y = labels

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
num_classes = len(labels.unique())
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

















def GNB():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb.score(X_test, y_test)

def svm():
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, preds)
    print("Accuracy using SVM:" ,accuracy)

def knn():
    neigh = KNeighborsClassifier(n_neighbors=3)
    
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,preds)
    print("Accuracy using Decision Tree:" ,accuracy)

def nn():

    input_dim = X_train.shape[1]  # Number of features
    
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    
    history = model.fit(X_train, y_train,
                        epochs=1000,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)
    
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    

def decisionTree():

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    #sns.heatmap(cf_matrix, annot=True)
    # Plot percentage of data
    
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')
    plt.figure()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print("Accuracy using Decision Tree:" ,accuracy)

def plotGenres():
    targetCounts = labels.value_counts()
    ax = sns.barplot(x=targetCounts.index, y=targetCounts.values, palette="Blues_d")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
plotGenres()    
