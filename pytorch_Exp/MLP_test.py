####################################################################################
#
#   Date: 2019.12.4, 12.16
#
#
#
####################################################################################

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 200)
        self.l2 = torch.nn.Linear(200, 10)

    def forward(self, input):
        x = input

        x = self.l1(x)
        x = self.l2(x)
        '''
        x = F.sigmoid(self.l1(x))
        x = self.l2(x)
        '''
        return x


class MadeDate:
    DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False
    DOWNLOAD_FSAHION_MNIST = False
    BATCH_SIZE = 50

    def __init__(self):
        # Mnist digits dataset
        '''
        if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
            # not mnist dir or mnist is empyt dir
            self.DOWNLOAD_MNIST = True
        if not (os.path.exists('./FashionMNIST/')) or not os.listdir('./FashionMNIST/'):
            # not mnist dir or mnist is empyt dir
            self.DOWNLOAD_FSAHION_MNIST = True
        '''

    def mnist(self):
        train_data = torchvision.datasets.MNIST(
            root='./mnist/',  # 保存或者提取位置
            train=True,  # this is training data
            transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
            # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
            download=self.DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
        )

        print('[mnist].train_data: ', train_data.train_data.size(), end='')  # (60000, 28, 28)
        print('.train_labels', train_data.train_labels.size())  # (60000)
        # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
        self.train_loader = Data.DataLoader(dataset=train_data,
                                            batch_size=self.BATCH_SIZE,
                                            shuffle=True)

        test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
        # 为了节约时间, 测试时只测试前2000个
        self.test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
            torch.FloatTensor
        )[:2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        # 设置DNA的size
        DNA.input_size = 28 * 28
        DNA.output_size = 10
        return self.train_loader, self.test_x, self.test_y

    def fashion_mnist(self):
        ''' fashion mnist 数据集 '''
        train_data = torchvision.datasets.FashionMNIST(
            root='./FashionMNIST/',  # 保存或者提取位置
            train=True,  # this is training data
            transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
            # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
            download=self.DOWNLOAD_FSAHION_MNIST,  # 没下载就下载, 下载了就不用再下了
        )
        # train_data.train_data = train_data.train_data.reshape(60000, 784)
        print('[fashion_mnist].train_data: ', train_data.train_data.size(),
              end='')  # (60000, 28, 28)
        print('.train_labels', train_data.train_labels.size())  # (60000)
        # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
        self.train_loader = Data.DataLoader(dataset=train_data,
                                            batch_size=self.BATCH_SIZE,
                                            shuffle=True)
        test_data = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=False)
        # 为了节约时间, 我们测试时只测试前2000个
        # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        self.test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
            torch.FloatTensor)[:2000] / 255.
        self.test_y = test_data.test_labels[:2000]

        return self.train_loader, self.test_x, self.test_y

    def getData(self):
        return self.train_loader, self.test_x, self.test_y


if __name__ == "__main__":
    data = MadeDate()
    train_loader, test_x, test_y = data.fashion_mnist()

    EPOCH = 10
    learning_rate = 0.1

    net = Model()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # the target label is not one-hotted
    loss_func = torch.nn.CrossEntropyLoss()

    test_x = test_x.view(-1, 784)
    print("->test_x: ", test_x.shape)
    accuracy = 0
    # training and testing
    for epoch in range(EPOCH):
        step = 0
        # TODO: 用movan的enumerate会报错，why?
        max_tep = int(60000 / train_loader.batch_size)
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(-1, 784)
            # print("[b_x, b_y].shape: ", b_x.shape, b_y.shape)
            # 分配 batch data, normalize x when iterate train_loader
            output = net(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            # clear gradients for this training step
            optimizer.zero_grad()
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 100 == 0:
                test_output = net(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float(
                    (pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                # print('Epoch: ', epoch, 'step: ', step,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
                print("\r" + 'Epoch: ' + str(epoch) + ' step: ' + str(step) + '[' +
                      ">>" * int(step / 50) + ']',
                      end=' ')
                print('loss: %.4f' % loss.data.numpy(), '| accuracy: %.2f' % accuracy, end=' ')
        print('')
    test_output = net(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number', '\n')
