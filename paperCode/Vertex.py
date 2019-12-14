
# The DNA holds Vertex and Edge instances. The Vertex class looks like this:


class Vertex(object):
    def __init__(self, vertex_proto):
        # Relationship to the rest of the graph.
        self.edges_in = set(vertex_proto.edges_in)  # Incoming edge IDs.
        self.edges_out = set(vertex_proto.edges_out)  # Outgoing edge IDs.

        # The type of activations.
        if vertex_proto.HasField('linear'):
            self.type = LINEAR  # Linear activations.
        elif vertex_proto.HasField('bn_relu'):
            self.type = BN_RELU  # ReLU activations with batch-normalization.
        else:
            raise NotImplementedError()

        # Some parts of the graph can be prevented from being acted upon by mutations.
        # The following boolean flags control this.
        self.inputs_mutable = vertex_proto.inputs_mutable
        self.outputs_mutable = vertex_proto.outputs_mutable
        self.properties_mutable = vertex_proto.properties_mutable
        # Each vertex represents a 2ˆs x 2ˆs x d block of nodes. s and d are positive
        # integers computed dynamically from the in-edges. s stands for "scale" so
        # that 2ˆx x 2ˆs is the spatial size of the activations. d stands for "depth",
        # the number of channels.

    def to_proto(self):
        ...
