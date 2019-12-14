
# For clarity, we omitted the details of a vertex ID targeting mechanism based on
# regular expressions, which is used to constrain(驱使,束缚) where the additional edges are placed.
# This mechanism ensured the skip connections only joined points in the “main
# convolutional backbone” of the convnet. The precedence range is used to give the main
# backbone precedence over the skip connections when resolving scale and depth conﬂicts
# in the presence of multiple incoming edges to avertex. Also omitted are details about
# the attributes of the edge to add.

# To evaluate an individual’s ﬁtness, its DNA is unfolded into a TensorFlow model by
# the Model class. This describes how each Vertex and Edge should be interpreted.
# For example:


class Model(object):
    ...

    def _compute_vertex_nonlinearity(self, tensor, vertex):
        """Applies the necessary vertex operations depending on the vertex type."""
        if vertex.type == LINEAR:
            pass
        elif vertex.type == BN_RELU:
            tensor = slim.batch_norm(
                inputs=tensor, decay=0.9, center=True, scale=True,
                epsilon=self._batch_norm_epsilon,
                activation_fn=None, updates_collections=None,
                is_training=self.is_training, scope='batch_norm')
            tensor = tf.maximum(tensor, vertex.leakiness * tensor, name='relu')
        else:
            raise NotImplementedError()
        return tensor

    def _compute_edge_connection(self, tensor, edge, init_scale):
        """Applies the necessary edge connection ops depending on the edge type."""
        scale, depth = self._get_scale_and_depth(tensor)
        if edge.type == CONV:
            scale_out = scale
            depth_out = edge.depth_out(depth)
            stride = 2 ** edge.stride_scale
            # ‘init_scale‘ is used to normalize the initial weights in the case of
            # multiple incoming edges.
            weights_initializer = slim.variance_scaling_initializer(
                factor=2.0 * init_scale ** 2, uniform=False)
            weights_regularizer = slim.l2_regularizer(
                weight=self._dna.weight_decay_rate)
            tensor = slim.conv2d(
                inputs=tensor, num_outputs=depth_out,
                kernel_size=[edge.filter_width(), edge.filter_height()],
                stride=stride, weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer, biases_initializer=None,
                activation_fn=None, scope='conv')
        elif edge.type == IDENTITY:
            pass
        else:
            raise NotImplementedError()
        return tensor


# The training and evaluation (Section 3.4) is done in a fairly standard way, similar
# to that in the tensorﬂow.org tutorials for image models. The individual’s ﬁtness is
# the accuracy on a held-out validation dataset, as described in the main text.

# Parents are able to pass some of their learned weights to their children (Section 3.6)
# . When a child is constructed from a parent, it inherits IDs for the different sets
# of trainable weights (convolution ﬁlters, batch norm shifts, etc.). These IDs are
# embedded(嵌入式的) in the TensorFlow variable names. When the child’s weights are initialized,
# those that have a matching ID in the parent are inherited, provided they have the
# same shape:
