
# The encoding for an individual is represented by a serializable DNA class instance containing all
# information except for the trained weights (Section 3.2). For all results in this paper, this
# encoding is a directed, acyclic graph where edges represent convolutions and vertices represent
# nonlinearities. This is a sketch of the DNA class:


class DNA(object):
    def __init__(self, dna_proto):
        """Initializes the ‘DNA‘ instance from a protocol buffer(协议缓冲区).
        The ‘dna_proto‘ is a protocol buffer used to restore the DNA state from disk.
        Together with the corresponding ‘to_proto‘ method, they allow for a
        serialization-deserialization(序列化-反序列化) mechanism.
        """
        # Allows evolving the learning rate, i.e. exploring the space of
        # learning rate schedules.
        # 编辑learning_rate
        self.learning_rate = dna_proto.learning_rate
        # 编辑结点、链接边
        self._vertices = {}  # String vertex ID to ‘Vertex‘ instance.
        for vertex_id in dna_proto.vertices:
            vertices[vertex_id] = Vertex(
                vertex_proto=dna_sproto.vertices[vertex_id])

        self._edges = {}  # String edge ID to ‘Edge‘ instance.
        for edge_id in dna_proto.edges:
            mutable_edges[edge_id] = Edge(edge_proto=dna_proto.edges[edge_id])

        ...

    def to_proto(self):
        """Returns this instance in protocol buffer form."""
        dna_proto = dna_pb2.DnaProto(learning_rate=self.learning_rate)

        for vertex_id, vertex in self._vertices.iteritems():
            dna_proto.vertices[vertex_id].CopyFrom(vertex.to_proto())

        for edge_id, edge in self._edges.iteritems():
            dna_proto.edges[edge_id].CopyFrom(edge.to_proto())
        ...
        return dna_proto

    def add_edge(self, dna, from_vertex_id, to_vertex_id, edge_type, edge_id):
        """Adds an edge to the DNA graph, ensuring internal consistency."""
        # ‘EdgeProto‘ defines defaults for other attributes.
        edge = Edge(EdgeProto(
            from_vertex=from_vertex_id, to_vertex=to_vertex_id, type=edge_type))
        self._edges[edge_id] = edge
        self._vertices[from_vertex_id].edges_out.add(edge_id)
        self._vertices[to_vertex].edges_in.add(edge_id)
        return edge
    # Other methods like ‘add_edge‘ to manipulate the graph structure.
    ...
