# 8.1
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:55:49 2018

@author: Harekrishna Panda
"""
"""
In [3]:
"""
# 8.1
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston

boston = load_boston()
bos = pd.DataFrame(boston.data)
price = pd.DataFrame(boston.target)

X =bos.iloc[:,:].values
y = price.iloc[:,:].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

intercept=regressor.intercept_
print(intercept)
coef=regressor.coef_
print(coef)
