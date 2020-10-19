import tensorflow as tf
import matplotlib.pyplot as plt
cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# 特征增强
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45, #随机旋转角度范围
    width_shift_range=.15, #随机水平偏移范围
    height_shift_range=.15, #随机垂直偏移范围
    horizontal_flip=True, #是否随机水平翻转
    zoom_range=[0.5,1] #比例缩放范围
)

class MyCNN(tf.keras.Model):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.CNN1 = tf.keras.Sequential(
            tf.keras.layers.Conv2D(filters=18,kernel_size=(3,3),strides=1,padding='same')
        )
        self.CNN2 = tf.keras.Sequential(
            tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), strides=1, padding='same')
        )
    def call(self, x):
        y = self.CNN1(x)
        print(y.shape)
        y = self.CNN2(x)
        print(y.shape)
        return y

model = MyCNN()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])

history = model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),epochs=5,validation_data=(x_test,y_test),validation_freq=1)

# 查看网络类型
model.summary()