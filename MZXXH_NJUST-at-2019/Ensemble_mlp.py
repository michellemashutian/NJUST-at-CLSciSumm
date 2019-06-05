#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-04 19:30
@desc: mlp by keras
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

def changeto2D(array_1D):
    result=[]
    for x in array_1D:
        y = 1 - x
        result.append(y)
    array=np.array(result)
    array_2D = [(i, j) for i, j in zip(array_1D, array)]
    output= np.array(array_2D)
    return output

loss="mean_squared_error"
             #mean_squared_error
             # categorical_crossentropy
             #sparse_categorical_crossentropy [0,1)
             # binary_crossentropy
optimizer="adam"     #sgd adam RMSprop
pred_probility_path=r"E:\\data\\subscribe\\vec\\subscribe\\w2v_ab-"+optimizer+"-"+loss+".txt"

x_test = np.loadtxt(r'E:\data\subscribe\vec\test\w2v_abstract_test.txt')
#y_test=np.loadtxt(r'E:\data\cv\test_label.txt')
#y_test = np.loadtxt(r'E:\data\cv\test_label.txt')
# y_test_1D = np.loadtxt(r'E:\data\cv\test_label.txt')
# y_test=changeto2D(y_test_1D)

x_train = np.loadtxt(r'E:\data\subscribe\vec\train\w2v_abstract_train.txt')
print(len(x_train))
y_train = np.loadtxt(r'E:\data\subscribe\vec\train\label1.txt')
# y_train_1D = np.loadtxt(r'E:\data\subscribe\vec\train\label1.txt')
# y_train=changeto2D(y_train_1D)
# 模型
model = Sequential()
model.add(Dense(64, input_dim=400, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))#softmax sigmoid
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=500)#d2v 32
#评估
y_predict = model.predict(x_test)
list_y_predict=y_predict.tolist()
# pred_true=0
f = open(pred_probility_path, "w", encoding="utf8")
for j in range(len(y_predict)):
   f.write(str(list_y_predict[j])+"\n")
#     if y_predict[j][0]>0.5:
#         y_predict[j][0]=1
#     else:
#         y_predict[j][0]=0
#     if  y_predict[j][0]==y_test[j][0]:
#         pred_true+=1
# zql=pred_true/len(y_predict)
# print(zql)

# keras 参数设置？