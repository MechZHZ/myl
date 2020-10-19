import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
'''
tensorflow2.0教程作业1：Regression 回归
练习二：
Tensorflow2.0，实现pm2.5预测，项目介绍详见hw1文件
来源：李宏毅2020机器学习作业1
'''

data = pd.read_csv('train.csv',encoding='gb18030')
# print(data.head())
data = data.iloc[:,3:]
data[data == 'NR'] = 0
data = np.array(data).astype(float)
# print(data.head())

x_data = []
y_data = []
for day in range(data.shape[0]//18):
    for hour in range(data.shape[1]):
        if day == 239 and hour > 14:
            break
        else:
            x_data.append([])
        for i in range(18):
            for j in range(9):
                if hour + j >= 24:
                    x_data[day * 24 + hour].append(data[(day + 1) * 18 + i][hour - 24 + j])
                else:
                    x_data[day * 24 + hour].append(data[day * 18 + i][hour + j])
        if hour > 14:
            y_data.append(data[(day + 1) * 18 + 9][hour + 9 - 24])
        else:
            y_data.append(data[day * 18 + 9][hour + 9])

x_data = np.array(x_data)
y_data = np.array(y_data)


learn_model = tf.keras.Sequential(tf.keras.layers.Dense(1,input_shape=(162,)))
learn_model.summary()
learn_model.compile(optimizer='adam',loss='mse')
history = learn_model.fit(x_data,y_data,epochs=100,validation_split=0.2,validation_freq=1)

# 绘图
plt.subplot(121)
plt.plot(history.history['loss'])
plt.subplot(122)
plt.plot(history.history['val_loss'])
plt.show()