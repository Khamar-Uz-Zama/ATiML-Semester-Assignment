# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:36:25 2020

@author: user
"""

import pandas as pd

data = pd.read_csv('data.csv', index_col=False)
data = data.drop(data.columns[0],1)
