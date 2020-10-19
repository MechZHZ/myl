import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
tensorflow2.0教程作业2：Classification 分类
练习三：
原生代码实现工资二分类，项目介绍详见hw2文件
来源：李宏毅2020机器学习作业2
'''


def main():
    '''
    数据预处理阶段
    '''
    # 读取数据
    with open("data/X_train") as f:
        next(f)  # 第一行不需要，所以从第二行开始
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)  # 第一列ID不需要，所以从1开始
        print(X_train)
    with open("data/Y_train") as f:
        next(f)  # 第一行不需要，所以从第二行开始
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)  # 第一列ID不需要，所以从1开始

    with open("data/X_test") as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

    # 归一化处理训练数据
    X_train, X_mean, X_std = normalize(X_train,Train = True)
    # 使用训练时的均值和方差处理归一化处理测试数据
    X_test, _, _ = normalize(X_test,Train = False, X_mean = X_mean, X_std = X_std)

    # 打乱各组数据顺序
    X_train,Y_train = shuffle(X_train,Y_train)

    # 10%的数据设置为验证集，其余为训练集
    dev_ratio = 0.1
    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

    '''
    数据处理完成，采用批量梯度下降法开始进行训练。
    '''
    # 总样本数量
    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    # 总参数数量
    train_dim = X_train.shape[1]
    # w为参数数量，b
    w = np.zeros((train_dim,))
    b = np.zeros((1,))

    # 迭代次数，每批次的数量，学习率,迭代次数
    max_iter = 100
    batch_size = 8
    l_r = 0.2
    step = 1

    # 记录数据用
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []

    # 训练
    for epoch in range(max_iter):
        # 每个epoch都会重新洗牌
        X_train, Y_train = shuffle(X_train, Y_train)

        # 分批次训练
        for idx in range(int(np.floor(train_size / batch_size))): #floor向下取整
            # 分批取样本
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

            # 计算梯度值
            w_grad, b_grad = gradient(X, Y, w, b)

            # 更新参数w和b
            # 学习率随着迭代时间增加而减少
            w = w - l_r / np.sqrt(step) * w_grad
            b = b - l_r / np.sqrt(step) * b_grad

            step = step + 1
        # 检测数据与损失
        y_train_pred = deal(X_train, w, b)
        # 四舍五入二分类问题
        Y_train_pred = np.round(y_train_pred)
        # 准确率记录
        train_acc.append(Accuracy(Y_train_pred, Y_train))
        # 交叉熵损失
        train_loss.append(Crossentropy_loss(y_train_pred, Y_train) / train_size)

        # 验证集
        y_dev_pred = deal(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(Accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(Crossentropy_loss(y_dev_pred, Y_dev) / dev_size)

        print('Training loss: {}'.format(train_loss[-1]))
        print('Development loss: {}'.format(dev_loss[-1]))
        print('Training accuracy: {}'.format(train_acc[-1]))
        print('Development accuracy: {}'.format(dev_acc[-1]))

    draw(train_loss,dev_loss,'loss',121)
    draw(train_acc, dev_acc,'acc',122)
    plt.show()


def draw(train,dev,name,a):
    # Loss curve
    plt.subplot(a)
    plt.plot(train)
    plt.plot(dev)
    plt.title(name)
    plt.legend(['train', 'dev'])


def gradient(X,Y,w,b):
    # 用sigmoid函数得预测值
    Y_pre = deal(X,w,b)
    Y_error = Y - Y_pre
    # 求取一批中的所有样本梯度
    w_grad = -np.sum(Y_error * X.T, 1)
    b_grad = -np.sum(Y_error)
    return w_grad,b_grad


def deal(X,w,b):
    return Sigmoid(np.matmul(X,w)+b)

def Sigmoid(z):
    # clip表示小于2值即为2，大于3值即为3
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))





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

# 区分训练集和验证集
def train_dev_split(X, Y, dev_ratio = 0.25):
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

#打乱数据顺序
def shuffle(X, Y):
    # 创建array[0 1 2 ... n]
    randomize = np.arange(len(X))
    # 打乱
    np.random.shuffle(randomize)
    # 返回打乱顺序后的数据
    return (X[randomize], Y[randomize])

# 准确率函数
def Accuracy(Y_pred, Y):
    acc = 1 - np.mean(np.abs(Y_pred - Y))
    return acc

# 交叉熵损失函数
def Crossentropy_loss(Y_pred, Y):
    cross_entropy = -np.dot(Y, np.log(Y_pred)) - np.dot((1 - Y), np.log(1 - Y_pred))
    return cross_entropy

if __name__ == '__main__':
    main()