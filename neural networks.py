# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:06:41 2020

@author: yazce
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
np.random.seed(2020) 

train_file_name = 'train.csv'
test_file_name = 'test.csv'

df_train = pd.read_csv(train_file_name)
df_test = pd.read_csv(test_file_name)
'''
df_train.info()
df_train.isnull().sum()
df_train.describe()
df_train.columns
df_test.info()
df_test.isnull().sum()
df_train.shape
df_test.shape

# Response目标字段
df_train['Response'].value_counts()
sns.countplot(df_train['Response'])

# As we can see most of the customers are between age 20 and 30
sns.distplot(df_train['Age'])
df_train.loc[(df_train['Age'] > 20) & (df_train['Age'] < 30)]['Age']
# 多维分析
plt.figure(figsize = (25,10))
sns.countplot(data = df_train,x = 'Age',hue = 'Response')

plt.figure(figsize = (10,8))
sns.countplot(data = df_train,x = 'Previously_Insured',hue = 'Response')

plt.figure(figsize = (10,5))
sns.countplot(data = df_train,x = 'Gender',hue = 'Response')

plt.figure(figsize = (15,7))
sns.scatterplot(y = 'Age',x = 'Annual_Premium',data = df_train)

plt.figure(figsize = (5,7))
sns.boxplot(data = df_train, y = 'Annual_Premium')

df_train['Vehicle_Age'].value_counts()
df_train.groupby(['Vehicle_Age','Response'])['Response'].count()
'''

#correlations = df_train.corr()
#f, ax = plt.subplots(figsize = (10, 10))
#sns.heatmap(correlations, annot = True)
df_train.head()   
from sklearn.preprocessing import LabelEncoder
def labelEncoder(data): 
    labelEncoder = LabelEncoder()   
    data['Gender'] = labelEncoder.fit_transform(data['Gender'])
    data['Vehicle_Damage'] = labelEncoder.fit_transform(data['Vehicle_Damage'])
    
    Vehicle_Age = ['< 1 Year','1-2 Year','> 2 Years']
    j = 0
    for i in Vehicle_Age:
        data['Vehicle_Age'] = data['Vehicle_Age'].replace(i, j)
        j += 1
    return data

df_train = labelEncoder(df_train)
df_test = labelEncoder(df_test)


tags = ['Gender', 'Age', 'Driving_License', 'Region_Code','Previously_Insured', 
        'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel', 
        'Vintage']

Standard_scaler = StandardScaler()
Standard_scaler1 = StandardScaler()

Standard_scaler.fit(df_train[tags].values)
x = Standard_scaler.transform(df_train[tags].values)

Standard_scaler1.fit(df_test[tags].values)
x_ = Standard_scaler1.transform(df_test[tags].values)
    
y = df_train['Response'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = keras.Sequential([
        keras.layers.Dense(512,activation='relu',input_shape=[10]), 
        keras.layers.Dense(256,activation='relu',input_shape=[10]),  
        keras.layers.Dense(128,activation='relu',input_shape=[10]),
        keras.layers.Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(x_train,y_train,batch_size = 500,epochs=100)

y_pred = model.predict(x_test)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

