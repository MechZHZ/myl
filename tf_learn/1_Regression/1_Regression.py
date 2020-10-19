import pandas as pd
import numpy as np
import math
from numpy.linalg import inv

'''
tensorflow2.0教程作业1：Regression 回归
练习一：
Python原生代码，实现pm2.5预测，项目介绍详见hw1文件
来源：李宏毅2020机器学习作业1
'''

def main():
#目标一：处理数据，得到学习用的X和Y
    #运用pandas读取数据，删除不要的行列：
    data = pd.read_csv('train.csv',encoding = 'gb18030')

    # 删除字列并将NP置为0
    data = data.iloc[0:,3:]
    data = data.replace(['NR'],[0])
    #或者写成data[data == 'NR'] = 0

    # 处理data变为数组
    # 去除了行头和列头并将字符变为浮点数变成数组pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
    data_new = data.to_numpy().astype(float)
    # 或者用data_new = np.array(data).astype(float)
    # print(data_new)
    # print(len(data_new))

    '''
    对data_new进行合并、拆分等处理,得到需要的X,Y
    每18X9的数据为X的一个元素，第十小时的pm2.5的为Y的一个元素，互相对应
    除了最后9个小时的数据不能录入x，其他都可以当作一组样本，共：240X24-9=5751组样本
    '''
    x,y= [],[]
    a = int(len(data_new)/18)
    print(len(data_new[0])-1)
    for day in range(0,int(len(data_new)/18)):
        for hour in range(len(data_new[0])):
            if day==239 and hour>14:
                break
            else:
                x.append([])
            # print(x)
            for i in range(18):
                for j in range(9):
                    if hour+j>=24:
                        x[day * 24 + hour].append(data_new[(day+1) * 18 + i][hour- 24 + j])
                    else:
                        x[day*24+hour].append(data_new[day*18+i][hour+j])
            if hour > 14:
                y.append(data_new[(day +1) * 18 + 9][hour+9-24])
            else:
                y.append(data_new[day*18+9][hour +9])
    x = np.array(x)
    print(x)
    # learning(x,y,10000)

'''
至此得到了需要的X,Y
下面对其进行训练
'''
def learning(x,y,times):
    # 系数矩阵w
    w = np.zeros(len(x[0]))
    # 转置x方便梯度计算
    xt = x.transpose()
    # 定义初始学习率lr
    lr = 0.1
    # 初始化梯度平方和
    s_gra = 0
    # 开始梯度下降
    for i in range(times):
        # 得到每次迭代的y值
        y_learn = np.dot(x,w)
        # 差值
        loss = y_learn - y
        # 计算总损失函数
        L = np.sum(loss**2)
        real_L = math.sqrt(L/len(y))
        # 利用Adagrad更新lr
        # 计算梯度，先求损失函数对w的微分，此处用数学计算得出：微分值为：2*loss*x
        # 转置是为了矩阵计算
        gra = 2*np.dot(xt,loss)
        # 得到Adagrad更新学习率的参数
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        # 更新参数矩阵w
        w = w - lr * gra / ada
        print('迭代次数:%d | 平均学习损失: %f ' % (i,real_L))
if __name__ == '__main__':
    main()