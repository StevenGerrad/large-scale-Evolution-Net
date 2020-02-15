import torchvision
import torch


class MadeData:
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

    def CIFR10(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 定义了我们的训练集，名字就叫trainset，至于后面这一堆，其实就是一个类：
        # torchvision.datasets.CIFAR10( )也是封装好了的，就在我前面提到的torchvision.datasets
        # 模块中,不必深究，如果想深究就看我这段代码后面贴的图1，其实就是在下载数据
        #（不翻墙可能会慢一点吧）然后进行变换，可以看到transform就是我们上面定义的transform
        trainset = torchvision.datasets.CIFAR10(root='./dataset',
                                                train=True,
                                                download=False,
                                                transform=transform)
        # trainloader其实是一个比较重要的东西，我们后面就是通过trainloader把数据传入网
        # 络，当然这里的trainloader其实是个变量名，可以随便取，重点是他是由后面的
        # torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块，
        #  网页链接http://pytorch.org/docs/0.3.0/data.html，这个类可见我后面图2
        self.trainloader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=self.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=2)
        # 对于测试集的操作和训练集一样，我就不赘述了
        testset = torchvision.datasets.CIFAR10(root='./dataset',
                                               train=False,
                                               download=False,
                                               transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=2000,
                                                      shuffle=False,
                                                      num_workers=2)
        # 设置DNA的size
        # DNA.input_size_height = 32
        # DNA.input_size_width = 32
        # DNA.input_size_channel = 3
        # DNA.output_size_height = 1
        # DNA.output_size_width = 1
        # DNA.output_size_channel = 10

        return self.trainloader, self.testloader

    def getData(self):
        return self.trainloader, self.testloader
