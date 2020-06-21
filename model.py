# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:36:25 2020

@author: Khamar Uz Zama

This document uses the extracted features to build various models.
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
from sklearn.ensemble import RandomForestClassifier

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
    """
    Gaussian Naive Bayes model
    """
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    accuracy = gnb.score(X_test, y_test)
    print("Accuracy using SVM:" ,accuracy)
    
    return accuracy

def SVM():
    """
    Support vector model
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, preds)
    print("Accuracy using SVM:" ,accuracy)

    return accuracy

def KNN():
    """
    K nearest Neighbors model
    """    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,preds)
    print("Accuracy using Decision Tree:" ,accuracy)
    
    return accuracy

def nn():
    """
    xxxx Remove nn? xxxx
    """
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
    """
    Decision Tree classifier
    """
    
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

    return accuracy

def randomForest():
    """
    Random Forest classifier
    """
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,y_train)    
    
    y_pred=rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    print("Accuracy using Random Forest:" ,accuracy)
    
    return accuracy

def plotGenres():
    """
    Plots the count of the genres counts of the data
    """
    targetCounts = labels.value_counts()
    ax = sns.barplot(x=targetCounts.index, y=targetCounts.values, palette="Blues_d")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

def plotAccuracies(accuracies):
    
    # this is for plotting purpose
    label = ["Decision Tree", "SVM", "Gaussian NB", "K-NN", "Random Forest"]
    index = np.arange(len(accuracies))
    plt.bar(index, accuracies)
    plt.xlabel('Model', fontsize=5)
    plt.ylabel('Accuracy', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()     

plotGenres()

dt = decisionTree()
sv = SVM()
nb = GNB()
kn = KNN()
rf = randomForest()

plotAccuracies([dt, sv, nb, kn, rf])

