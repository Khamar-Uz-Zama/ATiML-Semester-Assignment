# -*- coding: utf-8 -*-
"""
Created on Wed May 27 02:53:58 2020

@author: Khamar Uz Zama
"""

import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize 
from bs4 import BeautifulSoup


lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
pol = 0
stopwordsEN = set(stopwords.words('english'))


def readIndexes():    
    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    indexFile = 'master996.csv'
    
    indexPath = os.path.join(os.getcwd(), Path1, Path2, indexFile)
    data = pd.read_csv(indexPath, encoding='latin-1', sep=';')
    
    return data

def readHTMLFile(htmlFilePath):
    global pol
    
    try:
        with open(htmlFilePath, "r") as f:
            corpus = BeautifulSoup(f, features="lxml", from_encoding='utf-8').text
    except:
        print("cant read file",pol)
        pol+=1
        return False
    
    return corpus



def preProcessDocument(corpus, preProcessingConfig):
    
    processedSentences = []
    corpus = sent_tokenize(corpus)

    for sentence in corpus:
        wordList= []

        if(preProcessingConfig["lower"]):
            words=sentence.lower()
            
        if(preProcessingConfig["symbols"]):
            words = re.sub(r'[^a-zA-Z0-9 ]',r'',words)
        
        wordList= words.split()
        temp = wordList
        if(preProcessingConfig["lemmatize"]):
            for index, word in enumerate(temp):
                wordList[index] = lemmatizer.lemmatize(word)

        temp = wordList                
        if(preProcessingConfig["stem"]):
            for index, word in enumerate(temp):
                wordList[index] = stemmer.stem(words)
        
        if(preProcessingConfig["stopWords"]):
            for word in wordList:
                if word in stopwordsEN:
                    wordList.remove(word)
                
        processedSentences.append(" ".join(wordList))
        
    
    return processedSentences

def processHTMLFiles(numberOfFilesToRead, preProcessingConfig):

    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    HTMLFilesPath = 'Gutenberg_19th_century_English_Fiction'
    processedFiles = []
    badIndexes = []
    dataPath = os.path.join(os.getcwd(),Path1,Path2, HTMLFilesPath)
    data = readIndexes()
    if(numberOfFilesToRead < 0):
        numberOfFilesToRead = data.shape[0]
    labels = data['guten_genre'][:numberOfFilesToRead]
    
    for i in range(numberOfFilesToRead):
        print(i)
        htmlFilePath = os.path.join(dataPath,data['book_id'][i])[:-5] + '-content.html'
        corpus = readHTMLFile(htmlFilePath)
        if corpus:
            processed_corpus = preProcessDocument(corpus, preProcessingConfig)
            processedFiles.append(processed_corpus)
        else:
            badIndexes.append(i)
            
    labels = labels.drop(badIndexes)
    processedFiles.append(labels)
    
    if(len(badIndexes) > 0):
        print("Following files could not be read:")
        print(badIndexes)
        print("Total number of files dropped: ", len(badIndexes))
        
    return processedFiles