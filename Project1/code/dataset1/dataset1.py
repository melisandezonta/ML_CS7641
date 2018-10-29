#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 11:23:48 2017

@author: melisandezonta
"""

import numpy as np                                                     
import pandas as pd                                                    
import matplotlib.pyplot as plt  
import seaborn as sb 
from pandas.tools.plotting import scatter_matrix


# In[1]:
data = pd.read_csv('./dataset1/diabetes.csv') 
data.head(4)


# In[2]:
data = pd.DataFrame(data.values, columns = list(data))


# In[3]:
data.describe()


# In[4]:
l = list(data)
#sb.pairplot(data.dropna(), hue = l[-1])

# In[5]:
    
arr = data.values                                                      
arr_out = arr[:,-1]  
arr_in = arr[:,0:-1]
