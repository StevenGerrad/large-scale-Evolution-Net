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
                 stride_scale=0):
        self.from_vertex = from_vertex  # Source vertex ID.
        self.to_vertex = to_vertex  # Destination vertex ID.
        self.type = type

        # In this case, the edge represents a convolution.
        # 控制 channel 大小, this.channel = last channel * depth_factor
        self.depth_factor = depth_factor
        # Controls the strides(步幅) of the convolution. It will be 2ˆstride_scale. WHY ?????
        self.stride_scale = stride_scale

        if type == 'conv':
            # 卷积核 size
            # filter_width = 2 * filter_half_width + 1.
            self.filter_half_width = filter_half_width
            self.filter_half_height = filter_half_height
            # 定义卷积步长, 卷积步长必须是 2 的幂次方？

        # determine the inputs takes precedence in deciding the resolved depth or scale.
        # self.depth_precedence = edge_proto.depth_precedence
        # self.scale_precedence = edge_proto.scale_precedence

        self.input_channel = 0
        self.output_channel = 0

        # 用以继承权值
        self.model_id = 0


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

        # 用以继承权值
        # self.model_id = 0
