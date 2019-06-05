import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from keras import optimizers

train = np.loadtxt(r'.\train\scimm+train2018\feature_all.txt', dtype='float',encoding='utf-8')
train_label = np.loadtxt(r'.\train\scimm+train2018\label1_all.txt', dtype='int',encoding='utf-8')
#245468

test = np.loadtxt(r'.\test\feature_test.txt', dtype='float',encoding='utf-8')

model = Sequential()
model.add(Dense(128,input_dim=13,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))#softmax sigmoid
model.summary()

#binary_crossentropy categorical_crossentropy
optmr = optimizers.adam(lr=0.0005)
model.compile(optimizer=optmr,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train,train_label,epochs=10,batch_size=128,shuffle=True)
preResult = model.predict(test)
#1
thresold = round(0.55,2)# 0.55 0.65 0.75 0.85 0.95
final_result_pre = []  # 存储最后类别结果
for i in preResult:
    if i >= thresold:#如果概率>=thresold即认为为正例
        final_result_pre.append('1')
    else:
        final_result_pre.append('0')
with open(r'mlp_'+str(thresold)+r'.txt','w',encoding='utf-8') as f:
    f.writelines('\n'.join(final_result_pre))
print(thresold)
#2
thresold = round(0.65,2)# 0.55 0.65 0.75 0.85 0.95
final_result_pre = []  # 存储最后类别结果
for i in preResult:
    if i >= thresold:#如果概率>=thresold即认为为正例
        final_result_pre.append('1')
    else:
        final_result_pre.append('0')
with open(r'mlp_'+str(thresold)+r'.txt','w',encoding='utf-8') as f:
    f.writelines('\n'.join(final_result_pre))
print(thresold)
#3
thresold = round(0.75,2)# 0.55 0.65 0.75 0.85 0.95
final_result_pre = []  # 存储最后类别结果
for i in preResult:
    if i >= thresold:#如果概率>=thresold即认为为正例
        final_result_pre.append('1')
    else:
        final_result_pre.append('0')
with open(r'mlp_'+str(thresold)+r'.txt','w',encoding='utf-8') as f:
    f.writelines('\n'.join(final_result_pre))
print(thresold)
#4
thresold = round(0.85,2)# 0.55 0.65 0.75 0.85 0.95
final_result_pre = []  # 存储最后类别结果
for i in preResult:
    if i >= thresold:#如果概率>=thresold即认为为正例
        final_result_pre.append('1')
    else:
        final_result_pre.append('0')
with open(r'mlp_'+str(thresold)+r'.txt','w',encoding='utf-8') as f:
    f.writelines('\n'.join(final_result_pre))
print(thresold)
#5
thresold = round(0.95,2)# 0.55 0.65 0.75 0.85 0.95
final_result_pre = []  # 存储最后类别结果
for i in preResult:
    if i >= thresold:#如果概率>=thresold即认为为正例
        final_result_pre.append('1')
    else:
        final_result_pre.append('0')
with open(r'mlp_'+str(thresold)+r'.txt','w',encoding='utf-8') as f:
    f.writelines('\n'.join(final_result_pre))
print(thresold)
