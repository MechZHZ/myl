import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
'''
tensorflow2.0教程作业2：Classification 分类
练习五：
tensorflow2.0实现MNIST手写数字识别。
添加数据增强、断点续训、打印参数功能。
来源：清华tensorflow2.0教程
'''

class MyModel (tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Flatten()
        self.d2 = tf.keras.layers.Dense(128,input_shape=(784,),activation='relu')
        self.d3 = tf.keras.layers.Dense(10, input_shape=(128,), activation='softmax')
    def call(self,x):
        y = self.d1(x)
        y = self.d2(y)
        y = self.d3(y)
        return y

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 数据增强
x_train = x_train.reshape(x_train.shape[0],28,28,1)
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255., #比例缩放
    rotation_range=45, #随机旋转角度范围
    width_shift_range=.15, #随机水平偏移范围
    height_shift_range=.15, #随机垂直偏移范围
    horizontal_flip=False, #是否随机水平翻转
    zoom_range=[0.5,1] #比例缩放范围
)
image_gen_train.fit(x_train)

model = MyModel()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])


#断点续训
checkpoint_save_path = "./checkpoint/mnist.ckpt"
# 通过判断是否有索引表判断是否保存过
if os.path.exists(checkpoint_save_path + '.index'):
    print('---------load model ------------')
    model.load_weights(checkpoint_save_path)
# 只保留模型参数和最优参数
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True,save_best_only=True)

history = model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),epochs=5,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_callback])

# 查看网络类型
model.summary()

# 打印,保存参数
# 打印不省略
np.set_printoptions(threshold=np.inf)
print(model.trainable_weights)
file = open('../3_CNN/weights.txt', 'w')
for v in model.trainable_weights:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
# 绘图
plt.subplot(121)
plt.plot(history.history['loss'])
plt.subplot(122)
plt.plot(history.history['sparse_categorical_accuracy'])
plt.show()


