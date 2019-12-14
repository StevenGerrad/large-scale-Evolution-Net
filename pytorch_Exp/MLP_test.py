
####################################################################################
#
#   Date: 2019.12.4
#
#
#
####################################################################################

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 10)  # 不需要考虑batch_size
        self.l2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


if __name__ == "__main__":
    # 1. 数据准备
    # 保证使用from_numpy生成FloatTensor
    x_np = np.linspace(-1, 1, 100, dtype=np.float32)
    x = torch.unsqueeze(torch.from_numpy(x_np), dim=1)
    y = x ** 2 + 0.2 * torch.rand(x.size())
    x, y = Variable(x), Variable(y)
    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    # 2. 定义网络结构
    neural_net = Net()
    # print(neural_net) # 查看网络结构

    # 3. 训练网络
    optimizer = torch.optim.SGD(neural_net.parameters(), lr=0.5)
    loss_F = torch.nn.MSELoss()
    # 画图
    plt.ion()  # 打开交互模式,调用plot会立即显示,无需使用show()
    for t in range(100):
        # 重载了__call__()方法
        prediction = neural_net(x)  # 默认把第一维看成batch_size，定义网络时并不关心batch_size

        loss = loss_F(prediction, y)
        if t % 10 == 0:
            # 画图
            plt.cla()  # 清空图
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(),
                     color="red", linewidth=2.0)
            plt.text(0.5, 0.1, "loss: {:.5f}".format(loss.data.numpy()))
            plt.pause(0.1)  # 如果不暂停,循环10次会非常快。导致只能看到最后一张图

        optimizer.zero_grad()  # 因为每次反向传播的时候，变量里面的梯度都要清零
        loss.backward()  # 变量得到了grad
        optimizer.step()  # 更新参数
    plt.ioff()
    plt.show()  # 使用show()会阻塞(即，窗口不会变化，也不会自动退出)
