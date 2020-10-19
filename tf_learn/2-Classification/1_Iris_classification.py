from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
tensorflow2.0教程作业2：Classification 分类
练习一：
tensorflow2.0 原生代码，实现鸢尾花分类
来源：清华tensorflow2.0教程
'''
def main():
    # 读取sklearn数据集中的鸢尾花数据，改变x_data数据类型从float64到float32
    x_data = datasets.load_iris().data.astype(np.float32)
    y_data = datasets.load_iris().target
    # 或者用tf.cast(x_data,tf.float32转换数据类型)


    # 可选,归一化处理特征,大量特征时可消除取值范围的差异
    # x_mean = np.mean(x_data,axis=0)
    # x_std = np.std(x_data,axis=0)
    # # Z-score归一化数据
    # x_data = (x_data - x_mean) / (x_std + 1e-8)

    # shuffle数据，采用同一种子以配对
    np.random.seed(116)
    np.random.shuffle(x_data)
    np.random.seed(116)
    np.random.shuffle(y_data)
    tf.random.set_seed(116)

    # 区分训练集和验证集
    x_train = x_data[:-30]
    x_test = x_data[-30:]
    y_train = y_data[:-30]
    y_test = y_data[-30:]

    # 数据配对、打包，分批次训练准备
    train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

    '''
    按正态分布生成所有可学参数初始值
    4特征->3标签。一层，4*3 weight，3 bias
    '''
    w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
    b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

    # 学习参数初始化
    lr = 0.1
    train_loss_results =[]
    test_acc = []
    epochs = 200
    loss_all = 0
    # Adagrad记录平方梯度和
    # r_w1 = np.zeros([4,3])
    # r_b1 = np.zeros([3])
    # Adam参数
    # beta1, beta2, eps = 0.9, 0.999, 1e-6
    # mt1 = np.zeros([4,3])#动量法动量mt
    # mt2 = np.zeros([3])
    # vt1 = np.zeros([4,3])#RMSprop法小批量移动变量vt
    # vt2 = np.zeros([3])

    # 大训练轮数epoch
    for epoch in range(epochs):
        # 分批次训练
        for step,(x_train,y_train) in enumerate(train_db):
            # Gradient模型
            with tf.GradientTape() as tape:
                # 计算y值
                y = tf.matmul(x_train, w1)+b1
                # softmax函数归一结果，输出各类别的可能比例
                y = tf.nn.softmax(y)
                # 独热编码标签，三分类
                y_ = tf.one_hot(y_train,depth=3)
                # loss定义，均方差mse
                loss = tf.reduce_mean(tf.square(y_-y))
                # loss_all 记录每一轮所有batch的loss
                loss_all += loss.numpy()
            # 求loss对w1,b1的梯度
            grads = tape.gradient(loss,[w1,b1])
            # 更新参数w1,b1

            # SGD方法
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])

            # Adagrad 方法
            # r_w1 += tf.square(grads[0])
            # r_b1 += tf.square(grads[1])
            # w1.assign_sub(lr/tf.sqrt(r_w1) * grads[0])
            # b1.assign_sub(lr/tf.sqrt(r_b1) * grads[1])

            # Adam 方法
            # # 更新w1
            # vt1 = vt1 * beta1 + (1-beta1) * grads[0]
            # mt1 = mt1 * beta2 + (1-beta2) * tf.square(grads[0])
            # vt1_ = vt1 / (1 - beta1**(epoch+1))
            # mt1_ = mt1 / (1 - beta2**(epoch+1))
            # w1.assign_sub(lr*vt1_/(tf.sqrt(mt1_) + eps))
            # # 更新b1
            # vt2 = vt2 * beta1 + (1-beta1) * grads[1]
            # mt2 = mt2 * beta2 + (1-beta2) * tf.square(grads[1])
            # vt2_ = vt2/(1 - beta1**(epoch+1))
            # mt2_ = mt2 / (1 - beta2**(epoch+1))
            # b1.assign_sub(lr * vt2_ / (tf.sqrt(mt2_) + eps))

        print("Epoch{},loss{}".format(epoch,loss_all/4))
        # 每epoch更新loss_all
        train_loss_results.append(loss_all/4)
        loss_all = 0

        # 测试集测试
        total_correct, total_number = 0, 0
        for x_test, y_test in test_db:
            y = tf.matmul(x_test,w1) + b1
            y = tf.nn.softmax(y)
            # 最大概率的序号即为预测值,并更改数据类别
            pred = tf.argmax(y, axis=1)
            pred = tf.cast(pred, dtype=y_test.dtype)
            # 预测正确数量
            correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            # 记录一个epoch内所有batch正确数量和总数量
            total_correct += int(correct)
            total_number += x_test.shape[0]
        # 每一epoch的正确率
        acc = total_correct / total_number
        test_acc.append(acc)
        print('test_acc:', acc)


    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # $loss$ 为倾斜显示
    plt.plot(train_loss_results,label="$Loss$")
    plt.legend()
    plt.show()

    plt.title('Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(train_loss_results,label="$Accuracy$")
    plt.legend()
    plt.show()

# Z-score归一化函数
def normalize(X,Train = False, X_mean = False, X_std = False):
    if Train:
        #求取每个数据的平均值和标准差,函数中0表示为每列计算，reshape1，-1为转化为1行
        X_mean = np.mean(X, 0).reshape(1,-1)
        X_std  = np.std(X, 0).reshape(1,-1)
    #Z-score归一化数据，矩阵相减会自动分列或行
    X = (X - X_mean) / (X_std + 1e-8)
     #返回归一化后的数据，均值，标准差
    return X, X_mean, X_std

if __name__ == '__main__':
    main()