from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
tensorflow2.0教程作业2：Classification 分类
练习二：
tensorflow2.0  tf.keras实现鸢尾花分类
来源：清华tensorflow2.0教程
'''

# 获取数据
x_data = datasets.load_iris().data.astype(np.float32)
y_data = datasets.load_iris().target
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
'''
Sequential建立网络结构
拉直层:
tf.keras.layers.Flatten()
全连接层：
tf.keras.layers.Dense(activation=激活函数,kernel_regularizer=正则化名称)
'''
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(3,input_shape=(4,),activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())]
)

'''
建立Model类
'''

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.d1 = tf.keras.layers.Dense(3,input_shape=(4,),activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
#     def call(self, x):
#         y = self.d1(x)
#         return y
#
# model = MyModel()

'''
compile配置训练方法
optimizer = 'adam' 或 tf.keras.optimizers.Adam(lr=),loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False表示是否经过softmax函数)
metrics=['accuracy'数和数；'categorical_accuracy'独热码和独热码；'sparse_categorical_accuracy'y是独热码，y_是数]
'''
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])
'''
fit开始训练
validation_split=分割训练集比例
validation_freq=多少轮输出结果，history记录
'''
history = model.fit(x_data,y_data,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)
# 查看网络类型
model.summary()
# 绘图
plt.subplot(121)
plt.plot(history.history['loss'])
plt.subplot(122)
plt.plot(history.history['sparse_categorical_accuracy'])
plt.show()