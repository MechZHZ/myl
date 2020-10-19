import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
tensorflow2.0教程作业2：Classification 分类
练习四：
tensorflow2.0实现工资二分类，项目介绍详见hw2文件
来源：李宏毅2020机器学习作业2
'''

# 读取数据
with open("data/X_train") as f:
    next(f)  # 第一行不需要，所以从第二行开始
    x_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)  # 第一列ID不需要，所以从1开始
with open("data/Y_train") as f:
    next(f)  # 第一行不需要，所以从第二行开始
    y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)  # 第一列ID不需要，所以从1开始

with open("data/X_test") as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

# 归一化处理特征,大量特征时可消除取值范围的差异
x_mean = np.mean(x_train,axis=0)
x_std = np.std(x_train,axis=0)
# Z-score归一化数据
x_train = (x_train - x_mean) / (x_std + 1e-8)
# print(x_train.shape)

# shuffle数据，采用同一种子以配对
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

class MyModel (tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(2,input_shape=(510,),activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
    def call(self,x):
        y = self.d1(x)
        return y

model = MyModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.1),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train,y_train,batch_size=64,epochs=100,validation_split=0.2,validation_freq=20)

# 查看网络类型
model.summary()
# 绘图
plt.subplot(121)
plt.plot(history.history['loss'])
plt.subplot(122)
plt.plot(history.history['sparse_categorical_accuracy'])
plt.show()