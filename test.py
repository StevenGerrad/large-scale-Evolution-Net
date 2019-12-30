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
'''
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
'''

####################################################################################
#
#   class DNA --rubish 2019.12.17
#
####################################################################################
'''

class DNA(object):
    # __dna_cnt = 0
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10

    def __init__(self, learning_rate=0.05):
        global DNA_cnt
        self.dna_cnt = DNA_cnt
        DNA_cnt += 1

        self.fitness = -1.0

        self.learning_rate = learning_rate
        # layer
        self.vertices = []
        self.vertices.append(
            Vertex(edges_in=[],
                   edges_out=[0],
                   inputs_mutable=self.input_size,
                   outputs_mutable=self.input_size))
        self.vertices.append(
            Vertex(edges_in=[0],
                   edges_out=[1],
                   inputs_mutable=self.input_size,
                   outputs_mutable=self.hidden_size))
        self.vertices.append(
            Vertex(edges_in=[1],
                   edges_out=[],
                   inputs_mutable=self.hidden_size,
                   outputs_mutable=self.output_size))
        # edge
        self.edges = []
        self.edges.append(Edge(from_vertex=0, to_vertex=1))
        self.edges.append(Edge(from_vertex=1, to_vertex=2))

    def __del__(self):

        class_name = self.__class__.__name__
        print(class_name, "[", self.dna_cnt, "]销毁->fitness", self.fitness, end='\n')

    def add_edge(self, from_vertex_id, to_vertex_id, edge_type, edge_id):
        edge = Edge(from_vertex=from_vertex_id, to_vertex=to_vertex_id, type=edge_type)
        self.edges[edge_id] = edge
        self.vertices[from_vertex_id].edges_out.append(edge_id)
        self.vertices[to_vertex_id].edges_in.append(edge_id)
        return edge

    def calculate_flow(self):
        
        for vertex in self.vertices:
            # 先默认将input_mutable 置为0，然后处理inputs_mutable
            vertex.layer_size2zero()
            if len(vertex.edges_in) == 0:
                vertex.inputs_mutable = self.input_size
                # print("[calculate_flow]->start:")
            else:
                for i in vertex.edges_in:
                    vertex.inputs_mutable += self.vertices[
                        self.edges[i].from_vertex].outputs_mutable
                    # print("edge->",i,"from",self.edges[i].from_vertex,"to",self.edges[i].to_vertex,end=' ')
                    # print("vertex input: ", vertex.inputs_mutable)
        # outputs_mutable 默认不需要处理，即size
        

    def mutate_layer_size(self, v_list=[], s_list=[]):
        for i in range(len(v_list)):
            self.vertices[v_list[i]].outputs_mutable = s_list[i]

    def add_vertex(self, after_vertex_id, vertex_size, vertex_type):
        
        print(self.dna_cnt, "add_vertex", after_vertex_id)
        # 寻找上原 vertex 的前链接 edge
        last_edge_id = -1
        for edge_id in self.vertices[after_vertex_id].edges_in:
            if edge_id in self.vertices[after_vertex_id - 1].edges_out:
                last_edge_id = edge_id
                break
        # 修改原本位置的 edge, 并增加新的 edge (将所有新加入层后的 vertex_id++)
        for id, edge in enumerate(self.edges):
            if id == last_edge_id:
                continue
            if edge.from_vertex >= after_vertex_id:
                edge.from_vertex += 1
            if edge.to_vertex >= after_vertex_id:
                edge.to_vertex += 1
        self.edges.append(Edge(from_vertex=after_vertex_id, to_vertex=after_vertex_id + 1))
        # 将新加入的 vertex 插入(其实没有必要计算inputs_mutable)
        self.vertices.insert(
            after_vertex_id,
            Vertex(edges_in=[last_edge_id],
                   edges_out=[len(self.edges) - 1],
                   inputs_mutable=self.vertices[after_vertex_id - 1].outputs_mutable,
                   outputs_mutable=vertex_size,
                   type=vertex_type))
        for j, edge_id in enumerate(self.vertices[after_vertex_id + 1].edges_in):
            if self.edges[edge_id].from_vertex == after_vertex_id - 1 and self.edges[
                    edge_id].to_vertex == after_vertex_id + 1:
                self.vertices[after_vertex_id + 1].edges_in[j] = len(self.edges) - 1
                break
        print('')

    def has_edge(self, from_vertex_id, to_vertex_id):
        for edge_id in self.vertices[from_vertex_id].edges_out:
            if self.edges[edge_id].to_vertex == to_vertex_id:
                return True
        return False
'''

####################################################################################
#
#   class DNA 尝试的bfs便利检查整个网络
#
####################################################################################
'''
    def calculate_flow(self):     
        # 按顺序计算神经网络每层的输入输出size, outputs_mutable 默认不需要处理，即size
        
        # 先将所有 vertex 的 inputs_mutable 置 0
        for vertex in self.vertices:
            vertex.inputs_mutable = 0
        # 初始化记录edges节点状态的 done
        vis = [0] * len(self.edges)
        # 将vertices[0]的全部 edges_out 初始化
        stack = copy.deepcopy(self.vertices[0].edges_out)
        # 采用bfs搜索
        while len(stack) != 0:
            f = stack[0]
            if self.edges[f].state == 0:
                vis[f] = 1
                continue
            s = self.edges[f].from_vertex
            u = self.edges[f].to_vertex
            # 更新节点u的inputs_mutable TODO: 可以用list记录改vertex的所有edges_in是否都被处理过
            self.vertices[u].inputs_mutable += self.vertices[s].outputs_mutable
            vis[f] = 1
            del stack[0]
            for i in self.vertices[u].edges_out:
                if vis[i] == 0:
                    stack.append(i)
        # for vertex in self.vertices:
            # print("[calculate_flow].vertex: ", vertex.edges_in, vertex.inputs_mutable, vertex.edges_out,vertex.outputs_mutable)
    
    def add_vertex(self, before_vertex_id, after_vertex_id, vertex_size, vertex_type):
        # 简单版：所有 vertex 可排成一列 (但序号不一定按顺序)
        # 相当于隐藏了一个边,添加了一个节点两个边,并改变原边所对应的前后vertex的相关配置
        print(self.dna_cnt, "add_vertex", before_vertex_id)
        # 隐藏一个边
        for edge in self.edges:
            if edge.from_vertex == before_vertex_id and edge.to_vertex == after_vertex_id:
                edge.state = 0
        # 添加两个边
        new_vertex_id = len(self.vertices)
        self.edges.append(Edge(before_vertex_id, new_vertex_id))
        self.edges.append(Edge(new_vertex_id, after_vertex_id))
        # 添加了一个节点
        self.vertices.append(
            Vertex(edges_in=[len(self.edges) - 2],
                   edges_out=[len(self.edges) - 1],
                   inputs_mutable=self.vertices[before_vertex_id - 1].outputs_mutable,
                   outputs_mutable=vertex_size,
                   type=vertex_type))
        # 改变原边所对应的前后vertex的相关配置
        for i,edge_id in enumerate(self.vertices[before_vertex_id].edges_out):
            if self.edges[edge_id].from_vertex == before_vertex_id and self.edges[edge_id].to_vertex == after_vertex_id:
                self.vertices[before_vertex_id - 1].edges_out[i] = len(self.edges) - 2
        for i,edge_id in enumerate(self.vertices[after_vertex_id].edges_out):
            if self.edges[edge_id].from_vertex == before_vertex_id and self.edges[edge_id].to_vertex == after_vertex_id:
                self.vertices[before_vertex_id - 1].edges_out[i] = len(self.edges) - 1
        # print('')

'''

####################################################################################
#
#   test torch 模型的存储与读取
#
####################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt

x = np.random.randn(200)
y = np.random.randn(200)

plt.scatter(x, y, c='gray', marker='.')
plt.show()