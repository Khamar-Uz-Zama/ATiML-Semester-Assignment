# -*- coding: utf-8 -*-
"""
Created on Wed May 27 02:53:58 2020

@author: Khamar Uz Zama
"""

import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup

Path = 'Gutenberg_English_Fiction_1k\\Gutenberg_English_Fiction_1k'
indexPath = 'master996.csv'
filesPath = 'Gutenberg_19th_century_English_Fiction'

indexes = os.path.join(os.getcwd(),Path, indexPath)
dataPath = os.path.join(os.getcwd(),Path, filesPath)
data = pd.read_csv(indexes, encoding='latin-1', sep=';')

testData = data[:10]

htmlFile = os.path.join(dataPath,testData['book_id'][0])[:-5] + '-content.html'
with open(htmlFile, "r") as f:
    corpus = BeautifulSoup(f).text




from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize 

lmtzr = WordNetLemmatizer()
st = LancasterStemmer()

# Stores the stopwords in english in a set called stop
stopwordsEN = set(stopwords.words('english'))

def process_words(corp):
    
    # Initialising the list to store the stop words
    wordList= []
    wordLemmaList= []
    wordStemList = []
    
    
    # Converts all the uppercase in the Tweets to lowercase
    words=corp.lower()
    
    #Removing the special symbols such as ',','.', etc.
    words = re.sub(r'[^a-zA-Z0-9 ]',r'',words)
    
    #Converting stream of words to a list of words to check for stop words
    wordList= words.split()
    
    #Performing Lemmatisation
    for words in wordList:
        wordLemmaList.append(lmtzr.lemmatize(words))
        
    #Performing Word Stemming
    for words in wordList:        
        wordStemList.append(st.stem(words))
    
    #Removing the stopwords from the List of words and joins the words into string
    for words in wordStemList:
        if words in stop:
            wordStemList.remove(words)
    return(" ".join(wordStemList))

clean_text=[]
corp = sent_tokenize(corpus)
process_words(corp)

zz = [x.lower() for x in corp]


#Iterating through each row and proessing the text.
##We're saving this output in clean_text List
#for i in range(0, num_rows):
#    clean_text.append(process_words(str(textList[i])))    