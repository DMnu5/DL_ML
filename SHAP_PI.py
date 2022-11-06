# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:29:04 2022

@author: lenovo
"""
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense,Activation,Input,BatchNormalization,Dropout
from keras.models import Sequential,Model
from sklearn.metrics import r2_score as r2 
import scipy.stats
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
import warnings
from eli5.permutation_importance import get_score_importances
warnings.filterwarnings("ignore", category=Warning)
# ==========数据读取并加载================#

data= np.genfromtxt('C:/Users/lenovo/Desktop/data.csv',delimiter=',')  # 读取数据
np.random.shuffle(data)

Xtrain0=data[:240,1:-1]
ytrain0=data[:240,-1]
Xval0=data[240:320,1:-1]
yval0=data[240:320,-1]
Xtest0=data[320:400,1:-1]
ytest0=data[320:400,-1]

from sklearn import preprocessing
df = preprocessing.scale(data)
print(df.shape)

##划分数据集
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
    #model.add(BatchNormalization())
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))
    adam = Adam(lr = 0.0005)
    model.compile(optimizer= adam, loss='mse')
    return model

result_dir = 'C:/Users/lenovo/Desktop'
model = DNN()
model.fit(Xtrain,ytrain,validation_data=(Xval,yval),batch_size=128,epochs=500,shuffle=True)

def score(Xtrain, ytrain):
  y_pred = model.predict(Xtrain)
  return r2(ytrain,y_pred)
base_score, score_decreases = get_score_importances(score,Xtrain,ytrain)
feature_importances = np.mean(score_decreases, axis=0)
print(feature_importances)


import seaborn as sns
import shap
sns.set_style("white")
explainer =shap.KernelExplainer(model.predict,data=Xtrain)
shap.initjs()

shap_values = explainer.shap_values(Xtest)
shap.summary_plot(shap_values[0], Xtest)  #点图
shap.force_plot(explainer.expected_value, shap_values[0], Xtest.iloc[0,:])
