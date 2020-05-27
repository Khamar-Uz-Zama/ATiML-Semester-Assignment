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
    
    testData = data[:10]
    return testData

def readHTMLFile(htmlFilePath):
    with open(htmlFilePath, "r") as f:
        corpus = BeautifulSoup(f, features="lxml").text
    
    return corpus

def pre_process_document(corpus):
    
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

def processAllHTMLFiles(data):

    Path1 = 'Gutenberg_English_Fiction_1k'
    Path2 = 'Gutenberg_English_Fiction_1k'
    HTMLFilesPath = 'Gutenberg_19th_century_English_Fiction'

    dataPath = os.path.join(os.getcwd(),Path1,Path2, HTMLFilesPath)
    htmlFilePath = os.path.join(dataPath,data['book_id'][0])[:-5] + '-content.html'
    
    corpus = readHTMLFile(htmlFilePath)
    processed_corpus = pre_process_document(corpus)
    
    return processed_corpus
    
data = readIndexes()
pc = processAllHTMLFiles(data)