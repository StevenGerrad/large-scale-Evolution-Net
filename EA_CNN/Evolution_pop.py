import torch
import torch.nn.functional as F
import torch.utils.data as Data
from thop import profile

import os
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
import math

from MadeData import MadeData
from DNA import DNA
from StructMutation import StructMutation

# import global_var
from global_var import DNA_cnt


class Model(torch.nn.Module):
    def __init__(self, DNA, parent_model=None):
        super(Model, self).__init__()
        self.dna = DNA
        self.layer_vertex = torch.nn.ModuleList()
        # print('init vertex', end='')
        for i, vertex in enumerate(DNA.vertices):
            # print('v{} '.format(i), end='')
            # 默认第一层和最后一层 vertex 非 hidden 层
            if vertex.type == 'bn_relu':
                self.layer_vertex.append(
                    torch.nn.Sequential(torch.nn.BatchNorm2d(vertex.input_channel),
                                        torch.nn.ReLU(inplace=True)))
            elif vertex.type == 'Global Pooling':
                self.layer_vertex.append(
                    torch.nn.Sequential(
                        # torch.nn.AdaptiveAvgPool2d((1, 1)),
                        torch.nn.Linear(vertex.input_channel, DNA.output_size_channel)))
            else:
                self.layer_vertex.append(None)

        self.layer_edge = torch.nn.ModuleList()
        # print('\ninit edges', end='')
        for i, edge in enumerate(DNA.edges):
            # TODO: 默认padding补全
            # print('e{}:'.format(i), end='')
            if edge.type == 'conv':
                # print('{},{},{} |'.format(edge.filter_half_height, edge.filter_half_width,edge.stride_scale),end=' ')
                temp = torch.nn.Conv2d(edge.input_channel,
                                       edge.output_channel,
                                       kernel_size=(edge.filter_half_height * 2 + 1,
                                                    edge.filter_half_width * 2 + 1),
                                       stride=pow(2, edge.stride_scale),
                                       padding=(edge.filter_half_height, edge.filter_half_width))
                if edge.model_id != -1 or parent_model == None:
                    temp.weight = parent_model.layer_edge[i].weight
                self.layer_edge.append(temp)
            else:
                # print(end=' |')
                self.layer_edge.append(None)
        # print('')
        self.batch_size = Evolution_pop.BATCH_SIZE

    def forward(self, input):
        '''
        配置每层的 输入、输出、激活函数
        '''
        block_h = input.shape[0]
        x = {
            0: input,
        }
        for index, layer_vert in enumerate(self.layer_vertex[1:], start=1):
            length = len(x)

            a = torch.empty(block_h, 0, 0, 0)
            for j, edg in enumerate(self.dna.vertices[index].edges_in):
                ind_edg = self.dna.edges.index(edg)
                ind_x = self.dna.vertices.index(edg.from_vertex)
                t = x[ind_x]
                if edg.type == 'conv':
                    t = self.layer_edge[ind_edg](x[ind_x])
                if j == 0:
                    a = torch.empty(block_h, 0, t.shape[2], t.shape[3])
                a = torch.cat((a, t), dim=1)

            if self.dna.vertices[index].type == 'linear':
                x[index] = a
            elif self.dna.vertices[index].type == 'bn_relu':
                x[index] = layer_vert(a)
            elif self.dna.vertices[index].type == 'Global Pooling':
                temp = torch.nn.AdaptiveAvgPool2d((1, 1))
                a = temp(a)
                a = torch.squeeze(a, 3)
                a = torch.squeeze(a, 2)
                x[index] = layer_vert(a)

        return x[len(x) - 1]


class Evolution_pop:
    _population_size_setpoint = 10
    _evolve_time = 100
    fitness_pool = []

    EPOCH = 1  # 训练整批数据多少次
    BATCH_SIZE = 50
    N_CLASSES = 10

    # LR = 0.001          # 学习率

    def __init__(self, data, pop_max=10, evolve_time=100):
        '''
        初始化DNA: 一层hidden(节点数不同); 都为linear
        接收传入的训练数据 data
        初始化 Mutation 类
        '''
        self.population = []
        self.model_stack = {}

        for i in range(self._population_size_setpoint):
            dna_iter = DNA()
            self.population.append(dna_iter)
            dna_iter.calculate_flow()
            self.model_stack[dna_iter.dna_cnt] = Model(dna_iter)

            global DNA_cnt
            DNA_cnt = self._population_size_setpoint

        self.data = data
        self.struct_mutation = StructMutation()

        self._population_size_setpoint = pop_max
        self._evolve_time = evolve_time

        self.fitness_dir = {}

    def decode(self):
        '''
         对当前population队列中的每个未训练过的个体进行训练 
         https://www.cnblogs.com/denny402/p/7520063.html
        '''
        for dna in self.population:
            if dna.fitness != -1.0:
                continue
            # TODO: 新训练的个体将fitness加入fitness_pool
            # dna.calculate_flow()
            # net = Model(dna)

            net = self.model_stack[dna.dna_cnt]
            print("[decode].[", dna.dna_cnt, "]", net)

            optimizer = torch.optim.Adam(net.parameters(), lr=dna.learning_rate)
            # the target label is not one-hotted
            loss_func = torch.nn.CrossEntropyLoss()

            train_loader, testloader = self.data.getData()
            # print("[Evolution_pop].[decode]->test_x: ", test_x.shape)
            accuracy = 0
            # training and testing
            for epoch in range(self.EPOCH):
                step = 0
                # TODO: 用movan的enumerate会报错，why?
                max_tep = int(60000 / train_loader.batch_size)

                train_acc = .0
                len_y = 0
                for step, (b_x, b_y) in enumerate(train_loader):
                    # print("[b_x, b_y].shape: ", b_x.shape, b_y.shape)
                    # 分配 batch data, normalize x when iterate train_loader
                    output = net(b_x)  # cnn output
                    idy = b_y.view(-1, 1)
                    # b_y = torch.zeros(self.BATCH_SIZE, 10).scatter_(1, idy, 1).long()

                    loss = loss_func(output, b_y)  # cross entropy loss
                    # clear gradients for this training step
                    optimizer.zero_grad()
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    if step % 50 == 0:
                        pred = net(b_x)

                        print("\r" + 'Epoch: ' + str(epoch) + ' step: ' + str(step) + '[' +
                              ">>>" * int(step / 50) + ']',
                              end=' ')
                        # print('loss: %.4f' % loss.data.numpy(), '| accuracy: %.4f' % accuracy, end=' ')
                        print('loss: %.4f' % loss.data.numpy(), end=' ')
                print('')

            self.model_stack[dna.dna_cnt] = net
            # evaluation--------------------------------
            accuracy = self.Accuracy(net, testloader)
            input = torch.randn(self.BATCH_SIZE, dna.input_size_channel, dna.input_size_height,
                                dna.input_size_width)
            flops, params = profile(net, inputs=(input, ))
            print('----- Accuracy: {:.6f} Flops: {:.6f}-----'.format(accuracy, flops))
            # dna.fitness = eval_acc / len_y
            dna.fitness = accuracy
            self.fitness_dir[dna.dna_cnt] = accuracy
            print('')

    def Accuracy(self, net, testloader):
        ''' https://blog.csdn.net/Arctic_Beacon/article/details/85068188 '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        class_correct = list(0. for i in range(self.N_CLASSES))
        class_total = list(0. for i in range(self.N_CLASSES))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(self.BATCH_SIZE):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # for i in range(self.N_CLASSES):
        #     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        return sum(class_correct) / sum(class_total)

    def choose_varition_dna(self):
        '''
        每次挑选两个体,取fitness,判断要kill还是reproduce
        '''
        while self._evolve_time > 0:
            self._evolve_time -= 1
            self.decode()
            # 每次挑两个个体并提取出训练成绩fitness
            individual_pair = random.sample(list(enumerate(self.population)), 2)
            # TODO: 话说他这样取出来如果删掉的话真的能保证吗
            individual_pair.sort(key=lambda i: i[1].fitness, reverse=True)
            # better_individual = individual_pair[0]
            # worse_individual = individual_pair[1]
            # print("Choice: ",self._evolve_time,end=' ')
            # print("better: ",better_individual[0],'->',better_individual[1].fitness, end=' ')
            # print("worse: ", worse_individual[0],'->', worse_individual[1].fitness, end=' ')
            better_individual = individual_pair[0][0]
            worse_individual = individual_pair[1][0]
            individual_pair = []
            # (population过大->kill不好的)，反之(population过小->reproduce好的)
            if len(self.population) >= self._population_size_setpoint:
                print("--kill worse", worse_individual)
                self._kill_individual(worse_individual)
            elif len(self.population) < self._population_size_setpoint:
                print("--reproduce better", better_individual)
                self._reproduce_and_train_individual(better_individual)
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        print(self.population[0].fitness)
        self.population[0].calculate_flow()
        # self.pop_show()

    def _kill_individual(self, index):
        ''' kill by the index of population '''
        # self._print_population()

        del self.model_stack[self.population[index].dna_cnt]
        del self.population[index]

        # debug
        # self._print_population()

    def _reproduce_and_train_individual(self, index):
        ''' 
        inherit the parent, mutate, join the population 
        为了节省时间实际上有 Weight Inheritance
        '''
        # self._print_population()

        # inherit the parent (attention the dna_cnt)
        son = self.inherit_DNA(self.population[index])

        self.struct_mutation.mutate(son)
        son.calculate_flow()
        net = Model(son, self.model_stack[self.population[index].dna_cnt])

        self.model_stack[son.dna_cnt] = net
        self.population.append(son)
        # debug
        # self._print_population()

    def inherit_DNA(self, dna):
        ''' inderit from parent: reset dna_cnt, fitness '''
        son = copy.deepcopy(dna)
        global DNA_cnt
        son.dna_cnt = DNA_cnt
        DNA_cnt += 1
        son.fitness = -1
        return son

    def _print_population(self):
        print("pop sum: ", len(self.population), '|', end=' ')
        index = 0
        for i in self.population:
            print('(', index, '->', i.dna_cnt, ')', end=' ')
            index += 1
        print('')

    def pop_show(self):
        ''' 画出种群变化分布图 '''
        best_individual = self.population[0].dna_cnt
        live_individual = []
        for i in self.population:
            live_individual.append(i.dna_cnt)

        global DNA_cnt
        show_x = []
        show_y = []
        show_color = []
        for i in range(DNA_cnt + 1):
            if i in self.fitness_dir:
                show_x.append(i)
                show_y.append(self.fitness_dir[i])
                if i in live_individual:
                    if i == self.population[0].dna_cnt:
                        show_color.append('red')
                    else:
                        show_color.append('blue')
                else:
                    show_color.append('gray')
        plt.scatter(show_x, show_y, c=show_color, marker='.')
        plt.show()


if __name__ == "__main__":
    data = MadeData()
    # 数据集选择
    train_loader, testloader = data.CIFR10()

    # test = Evolution_pop(train_loader, test_x, test_y)
    test = Evolution_pop(data, pop_max=10, evolve_time=100)
    test.choose_varition_dna()
    test.pop_show()

    print()