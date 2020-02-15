from Vertex import Vertex
from Edge import Edge

# import global_var
from global_var import DNA_cnt
import random
import math


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
                    type='linear',
                    inputs_mutable=0,
                    outputs_mutable=0,
                    properties_mutable=0)
        # Global Pooling layer
        l1 = Vertex(edges_in=set(),
                    edges_out=set(),
                    type='linear',
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
        edg1 = Edge(from_vertex=l0, to_vertex=l1, type='identity')
        edg2 = Edge(from_vertex=l1, to_vertex=l2, type='identity')

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
                 depth_factor=1,
                 filter_half_width=None,
                 filter_half_height=None,
                 stride_scale=0):
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
        edge.model_id = -1

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

        for i, vertex in enumerate(self.vertices[1:], start=1):
            vertex.input_channel = 0
            print('vertex [', i, '].{}'.format(vertex.input_channel), end=' ')
            for edge in vertex.edges_in:
                edge.input_channel = edge.from_vertex.input_channel
                edge.output_channel = int(edge.input_channel * edge.depth_factor)
                vertex.input_channel += edge.output_channel

                f_ver = self.vertices.index(edge.from_vertex)
                if edge.type == 'identity':
                    f_h = 'N'
                else:
                    f_h = edge.filter_half_height
                if edge.type == 'identity':
                    f_w = 'N'
                else:
                    f_w = edge.filter_half_width
                print(', {}.{}_s{},{},{}'.format(f_ver, edge.type[0], edge.stride_scale, f_h, f_w),
                      end=' ')
            print()
        print('[calculate_flow] finish')

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
            depth_f = max(1.0, random.random() * 4)
            filter_h = 1
            filter_w = 1
            stride_s = math.floor(random.random() * 2)
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='conv',
                             depth_factor=depth_f,
                             filter_half_height=filter_h,
                             filter_half_width=filter_w,
                             stride_scale=0)
            edge_add1.model_id = -1
        else:
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='identity')
            edge_add1.model_id = -1
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
