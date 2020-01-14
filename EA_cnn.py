# The set contains the following mutations:

# • 学习率 ALTER-LEARNING-RATE (sampling details below).
# • 线性 IDENTITY (effectively means “keep training”).
# • 权重 RESET-WEIGHTS (sampled as in He et al. (2015), for
#       example).
# • 增 INSERT-CONVOLUTION (inserts a convolution at a random location in the “convolutional
#       backbone”, as in Figure 1. The inserted convolution has 3 × 3 filters, strides
#       of 1 or 2 at random, number of channels same as input.May apply batch-normalization
#       and ReLU activation or none at random).
# • 删 REMOVE-CONVOLUTION.
# • 步长 ALTER-STRIDE (only powers of 2 are allowed).
# • 通道数 ALTER-NUMBER-OF-CHANNELS (of random conv.).
# • 滤波器大小 FILTER-SIZE (horizontal or vertical at random, on random convolution, odd values only).
# • INSERT-ONE-TO-ONE (inserts a one-to-one/identity
#       connection, analogous to insert-convolution mutation).
# • 增跳层 ADD-SKIP (identity between random layers).
# • 删跳层 REMOVE-SKIP (removes random skip).

# 主要的组合实际为: conv+bn+relu

#########################################################################################################

# OUR--MLP

# 学习率α
# 增layer
# 增skip
# 设置'linear'/'relu'

# 设置layer:
#   out_channels
#   kernel_size
#   stride
#   padding

# 1.每次挑选两个个体, 确保个体已经被训练过了
# 2.挑选个体的fitness，(population过大->kill不好的)，反之(population过小->reproduce好的)

# Questiuon:
# 1. depth_factor 来决定channel, channel要是变小那么depth_factor是小数？若乘积结果不是整数需要取整
# 2. 仍要控制输出分类结果为one-hot
# 3. 是否默认维持 padding : 是

import os
import matplotlib.pyplot as plt
import random
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as Data

from torch.autograd import Variable
import torchvision

DNA_cnt = 0


class DNA(object):
    '''
    learning_rate, vertices[vertex_id]+type, edges[edge_id]
    由vertex(linear / bn_relu), 和 edge(conv / identity)组成
    '''
    # __dna_cnt = 0
    input_size_height = 32
    input_size_width = 32
    input_size_channel = 3

    output_size_height = 1
    output_size_width = 1
    output_size_channel = 10

    def __init__(self, learning_rate=0.05):
        '''
        注意，vertice 和 edges 中应该存引用
        '''
        global DNA_cnt
        self.dna_cnt = DNA_cnt
        DNA_cnt += 1
        self.fitness = -1.0
        self.learning_rate = learning_rate

        # input layer
        l0 = Vertex(edges_in=set(),
                    edges_out=set(),
                    type='identity',
                    inputs_mutable=0,
                    outputs_mutable=0,
                    properties_mutable=0)
        # Global Pooling layer
        l1 = Vertex(edges_in=set(),
                    edges_out=set(),
                    type='identity',
                    inputs_mutable=0,
                    outputs_mutable=0,
                    properties_mutable=0)
        # output layer
        l2 = Vertex(edges_in=set(),
                    edges_out=set(),
                    type='Global Pooling',
                    inputs_mutable=0,
                    outputs_mutable=0,
                    properties_mutable=0)
        self.vertices = []
        self.vertices.append(l0)
        self.vertices.append(l1)
        self.vertices.append(l2)

        # edge
        edg1 = Edge(from_vertex=l0, to_vertex=l1, type='linear')
        edg2 = Edge(from_vertex=l1, to_vertex=l2, type='linear')

        edg1.input_channel = self.input_size_channel
        edg1.output_channel = self.input_size_channel
        self.edges = []
        self.edges.append(edg1)
        self.edges.append(edg2)

        l0.edges_out.add(edg1)
        l1.edges_in.add(edg1), l1.edges_out.add(edg2)
        l2.edges_in.add(edg2)

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, "[", self.dna_cnt, "]销毁->fitness", self.fitness, end='\n')

    def add_edge(self,
                 from_vertex_id,
                 to_vertex_id,
                 edge_type='identity',
                 depth_factor=None,
                 filter_half_width=None,
                 filter_half_height=None,
                 stride_scale=None):
        """
        Adds an edge to the DNA graph, ensuring internal consistency.
        """
        edge = Edge(from_vertex=self.vertices[from_vertex_id],
                    to_vertex=self.vertices[to_vertex_id],
                    type=edge_type,
                    depth_factor=depth_factor,
                    filter_half_width=filter_half_width,
                    filter_half_height=filter_half_height,
                    stride_scale=stride_scale)
        self.edges.append(edge)
        self.vertices[from_vertex_id].edges_out.add(edge)
        self.vertices[to_vertex_id].edges_in.add(edge)
        return edge

    def calculate_flow(self):
        '''
        按顺序计算神经网络每层的输入输出参数
        '''
        self.vertices[0].input_channel = self.input_size_channel
        # self.vertices[0].output_channel = self.input_size_channel
        # self.vertices[-1].input_channel = self.output_size_channel
        # self.vertices[-1].output_channel = self.output_size_channel

        for vertex in self.vertices[1:]:
            for edge in vertex.edges_in:
                edge.input_channel = edge.from_vertex.input_channel
                edge.output_channel = int(edge.input_channel * edge.depth_factor)
                vertex.input_channel += edge.output_channel

    def mutate_layer_size(self, v_list=[], s_list=[]):
        for i in range(len(v_list)):
            self.vertices[v_list[i]].outputs_mutable = s_list[i]

    def add_vertex(self, after_vertex_id, vertex_type='linear', edge_type='identity'):
        '''
        3.0: 所有 vertex 和 edg 中记录的都是引用
        '''
        changed_edge = None
        # 先寻找那条应该被移除的边, 将其删除
        for i in self.vertices[after_vertex_id - 1].edges_out:
            if i.to_vertex == self.vertices[after_vertex_id]:
                self.vertices[after_vertex_id - 1].edges_out.remove(i)
                break
        for i in self.vertices[after_vertex_id].edges_in:
            if i.from_vertex == self.vertices[after_vertex_id - 1]:
                self.vertices[after_vertex_id].edges_in.remove(i)
                break
        for i, edge in enumerate(self.edges):
            if edge.from_vertex == self.vertices[
                    after_vertex_id - 1] and edge.to_vertex == self.vertices[after_vertex_id]:
                changed_edge = self.edges[i]

        # 创建新的 vertex, 并加入队列
        vertex_add = Vertex(edges_in=set(), edges_out=set(), type=vertex_type)
        self.vertices.insert(after_vertex_id, vertex_add)

        # 创建新的 edge, 并加入队列
        if edge_type == 'conv':
            depth_f = random.random() * 2
            filter_h = 1
            filter_w = 1
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='conv',
                             depth_factor=depth_f,
                             filter_half_height=filter_h,
                             filter_half_width=filter_w,
                             stride_scale=1)
        else:
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='linear')
        # 取代的那条边后移
        changed_edge.from_vertex = self.vertices[after_vertex_id]
        # edge_add2 = Edge(from_vertex=self.vertices[after_vertex_id],to_vertex=self.vertices[after_vertex_id + 1])
        self.edges.append(edge_add1)
        # self.edges.append(edge_add2)

        self.vertices[after_vertex_id - 1].edges_out.add(edge_add1)
        vertex_add.edges_in.add(edge_add1), vertex_add.edges_out.add(changed_edge)
        self.vertices[after_vertex_id + 1].edges_in.add(changed_edge)

    def has_edge(self, from_vertex_id, to_vertex_id):
        vertex_before = self.vertices[from_vertex_id]
        vertex_after = self.vertices[to_vertex_id]
        for edg in self.edges:
            if edg.from_vertex == vertex_before and edg.to_vertex == vertex_after:
                return True
        return False


class Vertex(object):
    '''
    edges_in, edges_out, HasField(bn_relu/linear), 
    inputs_mutable, outputs_mutable, properties_mutable
    '''
    def __init__(self,
                 edges_in,
                 edges_out,
                 type='linear',
                 inputs_mutable=1,
                 outputs_mutable=1,
                 properties_mutable=1):
        '''
        edges_in / edges_out : 使用set 
        each vertex can be inlear / 1*relu + 1*bn
        '''
        self.edges_in = edges_in
        self.edges_out = edges_out
        self.type = type  # ['linear' / 'bn_relu']

        self.inputs_mutable = inputs_mutable
        self.outputs_mutable = outputs_mutable
        self.properties_mutable = properties_mutable

        self.input_channel = 0
        # Each vertex represents a 2ˆs x 2ˆs x d block of nodes. s and d are positive
        # integers computed dynamically from the in-edges. s stands for "scale" so
        # that 2ˆx x 2ˆs is the spatial size of the activations. d stands for "depth",
        # the number of channels.


class Edge(object):
    '''
    No Need:type, depth_factor, filter_half_width, filter_half_height, 
            stride_scale, depth_precedence, scale_precedence
    '''
    def __init__(self,
                 from_vertex,
                 to_vertex,
                 type='identity',
                 depth_factor=1,
                 filter_half_width=None,
                 filter_half_height=None,
                 stride_scale=None):
        self.from_vertex = from_vertex  # Source vertex ID.
        self.to_vertex = to_vertex  # Destination vertex ID.
        self.type = type

        # In this case, the edge represents a convolution.
        # 控制 channel 大小, this.channel = last channel * depth_factor
        self.depth_factor = depth_factor
        if type == 'conv':
            # 卷积核 size
            # filter_width = 2 * filter_half_width + 1.
            self.filter_half_width = filter_half_width
            self.filter_half_height = filter_half_height
            # 定义卷积步长, 卷积步长必须是 2 的幂次方？
            # Controls the strides(步幅) of the convolution. It will be 2ˆstride_scale. WHY ?????
            self.stride_scale = stride_scale

        # determine the inputs takes precedence in deciding the resolved depth or scale.
        # self.depth_precedence = edge_proto.depth_precedence
        # self.scale_precedence = edge_proto.scale_precedence

        self.input_channel = 0
        self.output_channel = 0


class Model(torch.nn.Module):
    def __init__(self, DNA):
        super(Model, self).__init__()
        self.dna = DNA
        self.layer_vertex = torch.nn.ModuleList()
        for vertex in DNA.vertices:
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
        for edge in DNA.edges:
            # TODO: 默认padding补全
            if edge.type == 'conv':
                self.layer_edge.append(
                    torch.nn.Conv2d(edge.input_channel,
                                    edge.output_channel,
                                    kernel_size=(edge.filter_half_height * 2 + 1,
                                                 edge.filter_half_width * 2 + 1),
                                    stride=pow(2, edge.stride_scale),
                                    padding=(edge.filter_half_height, edge.filter_half_width)))
            else:
                self.layer_edge.append(None)
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

            if self.dna.vertices[index].type == 'identity':
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


class StructMutation():
    '''
    can mutate: hidden size, add edge, learning rate, add vertex, 
    '''
    def __init__(self):
        self._edge_types = []

    def mutate(self, dna):
        '''
        TODO: 可能出现由于概率'没有任何变异'的情况，不能让其发生
        1. 添加边时：添加identity, 则矩阵拼接时需要维度匹配 / 添加conv则需要是设置好参数
        '''
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # 1. Try the candidates in random order until one has the right connectivity.(Add)
        for from_vertex_id, to_vertex_id in self._vertex_pair_candidates(dna):
            if random.random() > 0.5:
                self._mutate_structure(mutated_dna, from_vertex_id, to_vertex_id)

        # 2. Try to mutate learning Rate
        self.mutate_learningRate(mutated_dna)

        # 3. mutate the hidden layer's size
        # self.mutate_hidden_size(dna)

        # 4. Mutate the vertex (Add)
        # self.mutate_vertex(mutated_dna)
        if random.random() > 0.4:
            self.mutate_vertex(mutated_dna)
        return mutated_dna

    def _vertex_pair_candidates(self, dna):
        """Yields connectable vertex pairs."""
        from_vertex_ids = self._find_allowed_vertices(dna)
        # if not from_vertex_ids: raise exceptions.MutationException(), 打乱次序
        random.shuffle(from_vertex_ids)

        to_vertex_ids = self._find_allowed_vertices(dna)
        # if not to_vertex_ids: raise exceptions.MutationException()
        random.shuffle(to_vertex_ids)

        for to_vertex_id in to_vertex_ids:
            # Avoid back-connections. TODO: 此处可能会涉及到 拓扑图的顺序判断
            # disallowed_from_vertex_ids, _ = topology.propagated_set(to_vertex_id)
            disallowed_from_vertex_ids = self._find_disallowed_from_vertices(dna, to_vertex_id)
            for from_vertex_id in from_vertex_ids:
                if from_vertex_id in disallowed_from_vertex_ids:
                    continue
                # This pair does not generate a cycle, so we yield it.
                yield from_vertex_id, to_vertex_id

    def _find_allowed_vertices(self, dna):
        ''' TODO: 除第一层(假节点)外的所有vertex_id '''
        return list(range(0, len(dna.vertices)))

    def _find_disallowed_from_vertices(self, dna, to_vertex_id):
        ''' 寻找不可作为起始层索引的：反向链接的，重复连接的Edge '''
        res = list(range(to_vertex_id, len(dna.vertices)))
        # 排查每个 vertex 是否不符合, 即索引在前面的 vertex 的所有 edges_out
        for i, vertex in enumerate(dna.vertices[:to_vertex_id]):
            for edge in vertex.edges_out:
                if dna.vertices.index(edge.to_vertex) == to_vertex_id:
                    if i not in res:
                        res.append(i)
                        continue
        return res

    def _mutate_structure(self, dna, from_vertex_id, to_vertex_id):
        """Adds the edge to the DNA instance."""
        if dna.has_edge(from_vertex_id, to_vertex_id):
            return False
        else:
            print("[_mutate_structure]->prepare to :", from_vertex_id, to_vertex_id)
            # TODO: edge 有两个类型，identity 和 conv (主要调节 stride, 在默认padding补全的情况下)
            # 1. 若数据维度不变，可以用identity， 则需要检查 stride 是否不变
            res = True
            bin_stride = 1
            for vertex_id, vert in enumerate(dna.vertices[from_vertex_id + 1:to_vertex_id]):
                edg_direct = vert.edges_in[0]
                for edg in vert.edges_in[1:]:
                    if edg.from_vertex == dna.vertices[
                            vertex_id - 1] and edg.to_vertex == dna.vertices[vertex_id]:
                        edg_direct = edg
                        break
                if edg_direct.stride != 1:
                    res = False
                    bin_stride *= edg_direct.stride
            if res:
                new_edge = dna.add_edge(from_vertex_id, to_vertex_id)
                return res
            # 2. 若数据维度改变(变小)，要用conv
            depth_f = random.random() * 2
            filter_h = 1
            filter_w = 1
            new_edge = dna.add_edge(from_vertex_id,
                                    to_vertex_id,
                                    edge_type='identity',
                                    depth_factor=depth_f,
                                    filter_half_height=filter_h,
                                    filter_half_width=filter_w,
                                    stride_scale=bin_stride)
            return True

    def mutate_hidden_size(self, dna):
        '''
        TODO: mutate the hidden layer's size 
        高斯分布随机生成, 对所有 hidden layer 变动...不可取
        '''
        # for i in list(range(1, len(dna.vertices) - 1)):
        #     if random.random() > 0.6:
        #         last = dna.vertices[i].outputs_mutable
        #         before = dna.vertices[i - 1].outputs_mutable
        #         after = dna.vertices[i + 1].outputs_mutable

        #         alpha = min(before - last, last - after) / 3
        #         next = last + alpha * np.random.randn(1)
        #         next = int(next[0])
        #         if next > before:
        #             next = before
        #         elif next < after:
        #             next = after
        #         dna.vertices[i].outputs_mutable = next

    def mutate_learningRate(self, dna):
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # Mutate the learning rate by a random factor between 0.5 and 2.0,
        # uniformly distributed in log scale.
        factor = 2**random.uniform(-1.0, 1.0)
        mutated_dna.learning_rate = dna.learning_rate * factor
        return mutated_dna

    def mutate_vertex(self, dna):
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # 随机选择一个 vertex_id 插入 vertex
        after_vertex_id = random.choice(self._find_allowed_vertices(dna))
        if after_vertex_id == 0:
            return mutated_dna

        print('outputs_mutable', dna.vertices[after_vertex_id].outputs_mutable,
              dna.vertices[after_vertex_id - 1].outputs_mutable)
        # TODO: how it supposed to mutate
        vertex_type = 'linear'
        if random.random() > 0.2:
            vertex_type = 'bn_relu'

        edge_type = 'identity'
        if random.random() > 0.2:
            edge_type = 'conv'

        mutated_dna.add_vertex(after_vertex_id, after_vertex_id, vertex_type, edge_type)
        return mutated_dna


class Evolution_pop:
    _population_size_setpoint = 4
    _max_layer_size = 3
    _evolve_time = 100
    fitness_pool = []

    EPOCH = 2  # 训练整批数据多少次
    BATCH_SIZE = 50
    N_CLASSES = 10

    # LR = 0.001          # 学习率

    def __init__(self, data):
        '''
        初始化DNA: 一层hidden(节点数不同); 都为linear
        接收传入的训练数据 data
        初始化 Mutation 类
        '''
        self.population = []
        for i in range(self._population_size_setpoint):
            dna_iter = DNA()
            self.population.append(dna_iter)
        self.data = data
        self.struct_mutation = StructMutation()

    def decode(self):
        '''
         对当前population队列中的每个未训练过的个体进行训练 
         https://www.cnblogs.com/denny402/p/7520063.html
        '''
        for dna in self.population:
            if dna.fitness != -1.0:
                continue
            # TODO: 新训练的个体将fitness加入fitness_pool
            dna.calculate_flow()
            net = Model(dna)
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

                    # target = torch.zeros(self.BATCH_SIZE, 10).scatter_(1, idy, 1).long()
                    # train_correct = (output == target).sum()
                    # train_acc += train_correct.data[0]
                    # len_y += len(target)

                    if step % 50 == 0:
                        pred = net(b_x)

                        # pred_y = torch.max(test_output, 1)[1].data.numpy()
                        # accuracy = float(
                        #     (pred_y == test_y.data.numpy()).astype(int).sum()) / float(
                        #         test_y.size(0))

                        # accuracy = self.Accuracy(net, testloader)
                        # print('Epoch: ', epoch, 'step: ', step,'| train loss: %.4f' % loss.data.numpy(),'| test accuracy: %.2f' % accuracy)
                        print("\r" + 'Epoch: ' + str(epoch) + ' step: ' + str(step) + '[' +
                              ">>>" * int(step / 50) + ']',
                              end=' ')
                        # print('loss: %.4f' % loss.data.numpy(), '| accuracy: %.4f' % accuracy, end=' ')
                        print('loss: %.4f' % loss.data.numpy(), end=' ')
                print('')

            # evaluation--------------------------------

            # net.eval()
            # eval_loss = 0.
            # eval_acc = 0.

            # len_y = 0
            # for batch_x, batch_y in testloader:
            #     batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y,
            #                                                                   volatile=True)
            #     out = net(batch_x)
            #     # b_y = torch.zeros(self.BATCH_SIZE, 10).scatter_(1, b_y, 1)

            #     loss = loss_func(out, batch_y)
            #     eval_loss += loss.data[0]
            #     pred = torch.max(out, 1)[1]

            #     target = torch.zeros(self.BATCH_SIZE, 10).scatter_(1, batch_y, 1).long()
            #     num_correct = (pred == target).sum()
            #     eval_acc += num_correct.data[0]

            #     len_y += len(target)
            # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len_y, eval_acc / len_y))

            accuracy = self.Accuracy(net, testloader)
            print('----- Accuracy: {:.6f} -----'.format(accuracy))
            # dna.fitness = eval_acc / len_y
            dna.fitness = accuracy
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

    def _kill_individual(self, index):
        ''' kill by the index of population '''
        # self._print_population()

        del self.population[index]
        # debug
        self._print_population()

    def _reproduce_and_train_individual(self, index):
        ''' 
        inherit the parent, mutate, join the population 
        为了节省时间实际上有 Weight Inheritance
        '''
        # self._print_population()

        # inherit the parent (attention the dna_cnt)
        son = self.inherit_DNA(self.population[index])

        self.struct_mutation.mutate(son)
        self.population.append(son)
        # debug
        self._print_population()

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
        DNA.input_size_height = 32
        DNA.input_size_width = 32
        DNA.input_size_channel = 3
        DNA.output_size_height = 1
        DNA.output_size_width = 1
        DNA.output_size_channel = 10
        return self.trainloader, self.testloader

    def getData(self):
        return self.trainloader, self.testloader


if __name__ == "__main__":
    data = MadeDate()
    # data.mnist()
    # 数据集选择
    # train_loader, test_x, test_y = data.getData()
    # train_loader, test_x, test_y = data.mnist()
    train_loader, testloader = data.CIFR10()

    # test = Evolution_pop(train_loader, test_x, test_y)
    test = Evolution_pop(data)
    test.choose_varition_dna()
