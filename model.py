# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:36:25 2020

@author: Khamar Uz Zama
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier




data = pd.read_csv('data.csv', index_col=False)
data = data.drop(data.columns[0],1)
X = data.iloc[:,:-1]
labels = data.iloc[:,-1]
y = labels


labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
#
#regressor = LinearRegression()
#regressor.fit(X_train,Y_train)
#y_preds = regressor.predict(X_test)


dt = DecisionTreeClassifier()


dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


cf_matrix = confusion_matrix(y_test, y_pred)


#sns.heatmap(cf_matrix, annot=True)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.figure()
plt.ylabel('True label')
plt.xlabel('Predicted label')






































def plotGenres():
    targetCounts = labels.value_counts()
    ax = sns.barplot(x=targetCounts.index, y=targetCounts.values, palette="Blues_d")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
plotGenres()    
