# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:04:51 2022

@author: lenovo
"""
import tensorflow
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense,Activation,Input,BatchNormalization,Dropout
from keras.models import Sequential,Model
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import r2_score#R square
import scipy.stats
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
import time
startTime = time.time()
from sklearn import preprocessing
import numpy as np
# np.random.seed(10)
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings("ignore", category=Warning)

import scipy.stats

# import data 
data= np.genfromtxt('C:/Users/lenovo/Desktop/repititon.csv',delimiter=',') # 读取数据
print(data.shape)

trainaverage = []
trainsd = []

for p in range(2,52,2):
    single_trainaverage = []
    
    for j in range(0,p):

        np.random.shuffle(data)
        data = preprocessing.scale(data)
            
        X=data[:,:-1]
        y=data[:,-1]
        
        X = X.astype('float32')
        y = y.astype('float32')
        
        def DNN():
            model = Sequential()
            model.add(Dense(150, input_shape=(X.shape[1],),activation='relu'))
            # model.add(Dense(150,activation='relu'))
            model.add(Dropout(0.25))
            model.add(BatchNormalization())
            model.add(Dense(200,activation='relu'))
            model.add(Dropout(0.35))
            model.add(BatchNormalization())
            model.add(Dense(100,activation='relu'))
            model.add(Dropout(0.25))
            # model.add(BatchNormalization())
            model.add(Dense(50,activation='relu'))
            model.add(Dense(1))
            adam = Adam(lr = 0.0005)
            model.compile(optimizer= adam, loss='mse')
            return model

        result_dir ='C:/Users/lenovo/Desktop/picuture' 
    
        model = DNN()
        model.fit(X,y,validation_split=0.1,batch_size=128,epochs=500,shuffle=True)
        
        D_pred1 = model.predict(X)
        
        D_pred1 =np.squeeze(D_pred1)
        y=np.squeeze(y)
        
        print(scipy.stats.pearsonr(y,  D_pred1)[0])
        single_trainaverage.append(scipy.stats.pearsonr(y,  D_pred1)[0])
   
    trainaverage.append(np.mean(single_trainaverage))
    trainsd.append(np.std(single_trainaverage))
    
print(trainaverage)
print(trainsd)