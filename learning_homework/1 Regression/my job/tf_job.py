import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


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

x_train = x_data[:3200,:]
x_test = x_data[3200:,:]
y_train = y_data[:3200]
y_test = y_data[3200:]
print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

learn_model = tf.keras.Sequential(tf.keras.layers.Dense(10,input_shape=(162,)))
learn_model.summary()
learn_model.compile(optimizer='adam',loss='mse')
history = learn_model.fit(x_train,y_train,epochs=100)