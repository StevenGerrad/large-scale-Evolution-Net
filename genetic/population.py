import numpy as np
import hashlib
import copy
import random
import math

from component import Edge, Vertex


class Individual(object):
    def __init__(self, params, indi_no):
        '''
        传入参数为json(字典)数据
        '''
        self.acc = -1.0  # 若结果非0, 在utils.load_population 中有处理
        self.id = indi_no  # for record the id of current individual
        self.number_id = 0  # for record the latest number of basic unit
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']

        t_vertices = params["vertices"]
        t_edges = params["edges"]

        self.vertices = []
        for ver in t_vertices:
            t_l = Vertex(edges_in=set(),
                         edges_out=set(),
                         type=ver["type"],
                         inputs_mutable=ver["inputs_mutable"],
                         outputs_mutable=ver["outputs_mutable"],
                         properties_mutable=ver["properties_mutable"])
            self.vertices.append(t_l)

        self.edges = []
        for edg in t_edges:
            t_e = Edge(from_vertex=self.vertices[edg["from_vertex"]],
                       to_vertex=self.vertices[edg["to_vertex"]],
                       type=edg["type"])
            self.edges.append(t_e)

        for i, ver in enumerate(self.vertices):
            for j in t_vertices[i]["edges_in"]:
                ver.edges_in.add(self.edges[j])
            for j in t_vertices[i]["edges_out"]:
                ver.edges_out.add(self.edges[j])
        # self.units = []

    def initialize(self):
        # initialize how many resnet unit/pooling layer/densenet unit will be used
        num_resnet = np.random.randint(self.min_resnet , self.max_resnet+1)
        num_pool = np.random.randint(self.min_pool , self.max_pool+1)
        num_densenet = np.random.randint(self.min_densenet, self.max_densenet+1)

        # find the position where the pooling layer can be connected
        # 随机排列resnet、pool、densenet的位置
        total_length = num_resnet + num_pool + num_densenet
        all_positions = np.zeros(total_length, np.int32)
        if num_resnet > 0: all_positions[0:num_resnet] = 1;
        if num_pool > 0: all_positions[num_resnet:num_resnet+num_pool] = 2;
        if num_densenet > 0 : all_positions[num_resnet+num_pool:num_resnet+num_pool+num_densenet] = 3;
        for _ in range(10):
            np.random.shuffle(all_positions)
        while all_positions[0] == 2: # pooling should not be the first unit
            np.random.shuffle(all_positions)

        # initialize the layers based on their positions
        input_channel = self.image_channel
        for i in all_positions:
            if i == 1:
                resnet = self.init_a_resnet(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None)
                input_channel = resnet.out_channel
                self.units.append(resnet)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)
            elif i == 3:
                densenet = self.init_a_densenet(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel)
                input_channel = densenet.out_channel
                self.units.append(densenet)

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

            for edge in vertex.edges_in:
                edge.input_channel = edge.from_vertex.input_channel
                edge.output_channel = int(edge.input_channel * edge.depth_factor)
                vertex.input_channel += edge.output_channel

    def __str__(self):
        _str=''
        for i, vertex in enumerate(self.vertices[1:], start=1):
            _str.join('vertex [', i, '].{}'.format(vertex.input_channel))
            for edge in vertex.edges_in:
                f_ver = self.vertices.index(edge.from_vertex)
                if edge.type == 'identity':
                    f_h = 'N'
                else:
                    f_h = edge.filter_half_height
                if edge.type == 'identity':
                    f_w = 'N'
                else:
                    f_w = edge.filter_half_width
                _str.join(', {}.{}_s{},{},{}'.format(f_ver, edge.type[0], edge.stride_scale, f_h, f_w))
            _str.join('\n')
        _str.join('[calculate_flow] finish\n')
        return '\n'.join(_str)

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


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


def test_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())


def test_population(params):
    pop = Population(params, 0)
    pop.initialize()
    print(pop)


if __name__ == '__main__':
    test_individual()
    #test_population()
