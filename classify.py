# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:01:33 2020

@author: Khamar Uz Zama
"""
import preProcessing as pp
import pickle
import seaborn as sns

filename = "processedHTML.pickle"
noOfFilesToLoad = 100
saveToPickle = False

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
    if(saveToPickle):
        with open(filename, 'wb') as f:
            pickle.dump(processedData,f)

z = data['guten_genre'].value_counts()
ax = sns.barplot(x=z.index, y=z.values, 
                 palette="Blues_d")
    
