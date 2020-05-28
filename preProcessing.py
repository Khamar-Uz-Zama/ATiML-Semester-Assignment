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

stopwordsEN = set(stopwords.words('english'))

def readIndexes():    
    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    indexFile = 'master996.csv'
    
    indexPath = os.path.join(os.getcwd(), Path1, Path2, indexFile)
    data = pd.read_csv(indexPath, encoding='latin-1', sep=';')
    
    return data
def readHTMLFile(htmlFilePath):
    try:
        with open(htmlFilePath, "r") as f:
            corpus = BeautifulSoup(f, features="lxml", from_encoding='utf-8').text
    except:
        pol+=1
        print("cant read",pol)
        return False
    return corpus

global pol
pol = 0

def preProcessDocument(corpus):
    
    processedSentences = []
    corpus = sent_tokenize(corpus)

    for sentence in corpus:
        wordList= []
        wordLemmaList= []
        wordStemList = []
        
        words=sentence.lower()
        
        words = re.sub(r'[^a-zA-Z0-9 ]',r'',words)
        
        wordList= words.split()
        
        for words in wordList:
            wordLemmaList.append(lemmatizer.lemmatize(words))
            
#        for words in wordList:        
#            wordStemList.append(stemmer.stem(words))
        
        for words in wordStemList:
            if words in stopwordsEN:
                wordLemmaList.remove(words)
                
        processedSentences.append(" ".join(wordLemmaList))
    
    return processedSentences

def processAllHTMLFiles(data, numberOfFilesToRead):

    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    HTMLFilesPath = 'Gutenberg_19th_century_English_Fiction'
    processedFiles = []

    dataPath = os.path.join(os.getcwd(),Path1,Path2, HTMLFilesPath)
    if(numberOfFilesToRead == -1):
        numberOfFilesToRead = data.shape[0]
    for i in range(numberOfFilesToRead):
        print(i)
        htmlFilePath = os.path.join(dataPath,data['book_id'][i])[:-5] + '-content.html'
        corpus = readHTMLFile(htmlFilePath)
        if corpus:
            processed_corpus = preProcessDocument(corpus)
            processedFiles.append(processed_corpus)
            
    return processedFiles