import os
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

def main():
    # 数据读取
    workspace_dir = './food-11'
    train_x, train_y = readfile(workspace_dir+'/training',True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(workspace_dir+"/validation",True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = readfile(workspace_dir+"/testing",False)
    print("Size of Testing data = {}".format(len(test_x)))


    # torch中的处理图片进行数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 随机将图片水平翻转
        transforms.RandomRotation(15), # 随机旋转图片
        transforms.ToTensor(), # 将图片向量化，并 normalize 到 [0,1] (data normalization)
    ])
    # testing 时不需要做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    # 打包数据集并分batch读取
    batch_size = 64
    train_set = ImgDataset(train_x, train_y, train_transform)
    val_set = ImgDataset(val_x, val_y, test_transform)
    # shuffle 为是否随机打乱参数
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = Classifier().cuda()
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
    num_epoch = 30  # 迭代次数

    # 训练
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0  # 计算每个opoch的精度与损失
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()  # 确保 model 是在 train model （开启 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 用 optimizer 将 model 参数的 gradient 归零
            train_pred = model(data[0].cuda())  # 利用 model 进行向前传播，计算预测值
            batch_loss = loss(train_pred, data[1].cuda())  # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
            batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新参数值
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            # 将結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                   train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
                   val_loss / val_set.__len__()))

def readfile(path,label):
    # 返回path所在文件夹下的所有文件
    print('reading [{}]...'.format(path))
    image_dir = sorted(os.listdir(path))
    # x存储所有图片
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    # y存储类别
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    # 读取所有图片
    for i, file in enumerate(image_dir):
        # 处理路径
        img = cv2.imread(os.path.join(path, file))
        # 存储所有图片，变为128*128
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          # 十种类别代表着不同种类，取前置数字表示
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

# 五层的CNN
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


if __name__ == '__main__':
    main()