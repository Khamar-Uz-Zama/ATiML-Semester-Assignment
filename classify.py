# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:01:33 2020

@author: Khamar Uz Zama
"""
import preProcessing as pp

data = pp.readIndexes()
pc = pp.processAllHTMLFiles(data,100)
