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
