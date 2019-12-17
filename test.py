####################################################################################
#
#   show the random scatter --plt
#
####################################################################################
'''
import torch
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)  # 数据的基本形态
x0 = torch.normal(0.25 * n_data, .08)
y0 = torch.zeros(100)
x1 = torch.normal(0.5 * n_data, .08)
y1 = torch.ones(100)
x2 = torch.normal(0.75 * n_data, .08)
y2 = torch.ones(100) * 2

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
# FloatTensor = 32-bit floating
x = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)
# LongTensor = 64-bit integer
y = torch.cat((y0, y1, y2), ).type(torch.LongTensor)

print(x.data.numpy().shape, y.data.numpy().shape[0])
# plt.scatter(x.data.numpy(), y.data.numpy())
plt.scatter(x.data.numpy()[:, 0],
            x.data.numpy()[:, 1],
            c=y.data.numpy(),
            s=100,
            lw=0,
            cmap='RdYlGn')
plt.show()
'''
####################################################################################
#
#   test pytorch from movan
#
####################################################################################
'''
import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.l1 = torch.nn.Linear(5, 5)
        self.l2 = torch.nn.Linear(5, 5)
        self.l3 = torch.nn.Linear(10, 5)
        self.l4 = torch.nn.Linear(10, 5)

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        # x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        # x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x1 + x2)
        x4 = self.l4(x2 + x3)
        return x

net = Net() # 几个类别就几个 output

print(net)  # net 的结构

'''
'''
Net(
  (l1): Linear(in_features=5, out_features=5, bias=True)
  (l2): Linear(in_features=10, out_features=5, bias=True)
)
'''
'''
import random
class A():
    def __init__(self):
        self.a = random.randint(0, 50)


class B():
    def __init__(self):
        self.b = []
        for i in range(10):
            temp = A()
            self.b.append(temp)

    def print_b(self):
        for i in self.b:
            print('{%d},' % (i.a), end='')
        print(' ')


if __name__ == "__main__":
    b = B()
    b.print_b()
    # 取出再销毁
    sub2 = b.b[2 - 1]
    print(sub2.a)
    del sub2
    b.print_b()
    # 将原列表中的引用销毁
    del b.b[2 - 1]
    b.print_b()
'''

####################################################################################
#
#   try enumerate
#
####################################################################################
'''
import random

A = ['A', 'B', 'C', 'D']
a = list(enumerate(A))
c = random.sample(a, 2)
print(c)
'''

####################################################################################
#
#   try 打印进度条
#
####################################################################################
'''
import time


def Cus_precess(max_tep):
    for i in range(1, max_tep):
        print('[' + '>' * i + ' ' * (max_tep - i) + ']' + str(int(100 / (max_tep - 1) * i)) + '%' +
              '\r',
              end='')
        time.sleep(0.2)
    print('\n')


def main():
    # max = int(input('最大步数：'))
    max = 100
    Cus_precess(max)


if __name__ == '__main__':
    main()

'''

####################################################################################
#
#   float 转换为string并插入‘.’
#
####################################################################################
'''
temp_str = str(int(accuracy * 1e5))
temp_l = list(str(int(accuracy * 1e5)))
temp_l.insert(-3, '.')
show_acc = "".join(temp_l)
'''

####################################################################################
#
#   torch tensor + - contact
#
####################################################################################
'''
import torch

a = torch.ones(3, 4) * 3
print(a)
b = torch.ones(3, 4) * 2
c = a + b
print(c)

# stack 是建立一个新的维度
d = torch.stack((a, b), dim=1)
print(d, d.shape)
e = torch.stack((a, b), dim=0)
print(e, e.shape)

# cat
f = torch.cat((a, b), dim=0)
print(f, f.shape)
g = torch.cat((a, b), dim=1)
print(g, g.shape)

h = torch.empty(3,0)
print(h)
h = torch.cat((h, a), dim=1)
print(h)

'''

####################################################################################
#
#   torch tensor + - contact
#
####################################################################################

import numpy as np
import matplotlib.pyplot as plt

# size = 10000
# a = np.random.normal(loc=3, scale=1, size=size)
# # plt.bar(list(range(0, size)), a)
# plt.hist(a)
# plt.show()
# print(a)

# m1=100
# sigma=20
# x = m1 + sigma * np.random.randn(2000)
x = np.random.randn(1)
plt.hist(x, bins=50, color="green", normed=True)
plt.show()
