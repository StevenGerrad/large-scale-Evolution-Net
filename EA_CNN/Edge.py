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