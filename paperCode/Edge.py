
class Edge(object):

    def __init__(self, edge_proto):
        # Relationship to the rest of the graph.
        self.from_vertex = edge_proto.from_vertex  # Source vertex ID.
        self.to_vertex = edge_proto.to_vertex  # Destination vertex ID.
        if edge_proto.HasField('conv'):
            # In this case, the edge represents a convolution.
            self.type = CONV

            # Controls the depth (i.e. number of channels) in the output, relative to the
            # input. For example if there is only one input edge with a depth of 16 channels
            # and ‘self._depth_factor‘ is 2, then this convolution will result in an output
            # depth of 32 channels. Multiple-inputs with conflicting depth must undergo
            # depth resolution first.
            self.depth_factor = edge_proto.conv.depth_factor

            # Control the shape of the convolution filters (i.e. transfer function).
            # This parameterization ensures that the filter width and height are odd
            # numbers: filter_width = 2 * filter_half_width + 1.
            self.filter_half_width = edge_proto.conv.filter_half_width
            self.filter_half_height = edge_proto.conv.filter_half_height

            # Controls the strides(步幅) of the convolution. It will be 2ˆstride_scale.
            # Note that conflicting input scales must undergo scale resolution. This
            # controls the spatial scale of the output activations relative to the
            # spatial scale of the input activations.
            self.stride_scale = edge_proto.conv.stride_scale
        elif edge_spec.HasField('identity'):
            self.type = IDENTITY
        else:
            raise NotImplementedError()

        # In case depth or scale resolution is necessary due to conflicts in inputs,
        # These integer parameters determine which of the inputs takes precedence in
        # deciding the resolved depth or scale.
        self.depth_precedence = edge_proto.depth_precedence
        self.scale_precedence = edge_proto.scale_precedence

    def to_proto(self):
        ...
