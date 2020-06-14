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
#from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data.csv', index_col=False)
data = data.drop(data.columns[0],1)
X = data.iloc[:,:-1]
labels = data.iloc[:,-1]
y = labels

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
num_classes = len(labels.unique())
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU



cnnModel = Sequential()
cnnModel.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
cnnModel.add(LeakyReLU(alpha=0.1))
cnnModel.add(MaxPooling2D((2, 2),padding='same'))
cnnModel.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
cnnModel.add(LeakyReLU(alpha=0.1))
cnnModel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
cnnModel.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
cnnModel.add(LeakyReLU(alpha=0.1))                  
cnnModel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
cnnModel.add(Flatten())
cnnModel.add(Dense(128, activation='linear'))
cnnModel.add(LeakyReLU(alpha=0.1))                  
cnnModel.add(Dense(num_classes, activation='softmax'))































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
