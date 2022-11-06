# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:58:59 2022

@author: lenovo
"""
import tensorflow
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense,Activation,Input,BatchNormalization,Dropout
from keras.models import Sequential,Model
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score#R square
import scipy.stats
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
import time
startTime = time.time()
import numpy as np
# np.random.seed(10)

import warnings
warnings.filterwarnings("ignore", category=Warning)

import scipy.stats
# ==========数据读取并加载================#

data= np.genfromtxt('C:/Users/lenovo/Desktop/data.csv',delimiter=',')  # 读取数据

R_Trainresult = []
R_Valresult = []
R_Testresult = []
R2_Trainresult = []
R2_Valresult = []
R2_Testresult = []
MSE_Trainresult = []
MSE_Valresult = []
MSE_Testresult = []
MAE_Trainresult = []
MAE_Valresult = []
MAE_Testresult = []
for j in range(0,20):
    np.random.shuffle(data)
    
    Xtrain0=data[:240,1:-1]
    ytrain0=data[:240,-1]
    Xval0=data[240:320,1:-1]
    yval0=data[240:320,-1]
    Xtest0=data[320:400,1:-1]
    ytest0=data[320:400,-1]

    ##标准化
    from sklearn import preprocessing
    df = preprocessing.scale(data)

    Xtrain=df[:240,1:-1]
    ytrain=df[:240,-1]
    Xval=df[240:320,1:-1]
    yval=df[240:320,-1]
    Xtest=df[320:400,1:-1]
    ytest=df[320:400,-1]


    Xtrain = Xtrain.astype('float32')
    ytrain = ytrain.astype('float32')
    Xval = Xval.astype('float32')
    yval = yval.astype('float32')
    Xtest = Xtest.astype('float32')
    ytest = ytest.astype('float32')

# ==========模型建立及求解================#
#####################模型搭建核心代码，其中的激活函数等都可调###############################
    def DNN():
        model = Sequential()
        model.add(Dense(150, input_shape=(Xtrain.shape[1],),activation='relu'))
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
    model.fit(Xtrain,ytrain,validation_data=(Xval,yval),batch_size=128,epochs=500,shuffle=True)
    # plot_history(history, result_dir)  
    D_pred1 = model.predict(Xtrain)
    
    R2_Trainresult.append(r2_score(ytrain, D_pred1))
    MSE_Trainresult.append(mean_squared_error(ytrain, D_pred1))
    MAE_Trainresult.append(mean_absolute_error(ytrain, D_pred1))
#####################################################################
    D_pred2 = model.predict(Xval)
    
    R2_Valresult.append(r2_score(yval, D_pred2))
    MSE_Valresult.append(mean_squared_error(yval, D_pred2))
    MAE_Valresult.append(mean_absolute_error(yval, D_pred2))
   
   
##################################################################################################
    D_pred = model.predict(Xtest)
    
    R2_Testresult.append(r2_score(ytest, D_pred))
    MSE_Testresult.append(mean_squared_error(ytest, D_pred))
    MAE_Testresult.append(mean_absolute_error(ytest, D_pred))
    
    ytrain=np.squeeze(ytrain)
    D_pred1 =np.squeeze(D_pred1)
    yval=np.squeeze(yval)
    D_pred2 =np.squeeze(D_pred2)
    ytest= np.squeeze(ytest)
    D_pred =np.squeeze(D_pred)
    R_Trainresult.append(scipy.stats.pearsonr(ytrain, D_pred1)[0])
    R_Testresult.append(scipy.stats.pearsonr(ytest, D_pred)[0])
    R_Valresult.append(scipy.stats.pearsonr(yval, D_pred2)[0])