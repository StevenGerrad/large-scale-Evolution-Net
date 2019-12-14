
# FLOPs estimation

# This section describes how we estimate the number of ﬂoating point operations (FLOPs)
# required for an entire evolution experiment. To obtain the total FLOPs, we sum the FLOPs
# for each individual ever constructed. An individual’s FLOPs are the sum of its training
# and validation FLOPs. Namely, the individual FLOPs are given by FtNt + FvNv, where Ft is
# the FLOPs in one training step, Nt is the number of training steps, Fv is the FLOPs
# required to evaluate one validation batch of examples and Nv is the number of validation
# batches.

# The number of training steps and the number of validation batches are known in advance
# and are constant throughout the experiment. Ft was obtained analytically as the sum of
# the FLOPs required to compute each operation executed during training (that is, each
# node in the TensorFlow graph). Fv was found analogously.

# Below is the code snippet that computes FLOPs for the training of one individual,
# for example.

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
import tensorflow as tf
tfprof_logger = tf.contrib.tfprof.python.tools.tfprof.tfprof_logger


def compute_flops():
    """Compute flops for one iteration of training."""
    graph = tf.Graph()
    with graph.as_default():
        # Build model
        ...
        # Run one iteration of training and collect run metadata.
        # This metadata will be used to determine the nodes which were
        # actually executed as well as their argument shapes.
        run_meta = tf.RunMetadata()
        with tf.Session(graph=graph) as sess:
            feed_dict = {...}
            _ = sess.run(
                [train_op],
                feed_dict=feed_dict,
                run_metadata=run_meta,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))

        # Compute analytical FLOPs for all nodes in the graph.
        logged_ops = tfprof_logger._get_logged_ops(
            graph, run_meta=run_metadata)

        # Determine which nodes were executed during one training step
        # by looking at elapsed execution time of each node.
        elapsed_us_for_ops = {}
        for dev_stat in run_metadata.step_stats.dev_stats:
            for node_stat in dev_stat.node_stats:
                name = node_stat.node_name
                elapsed_us = node_stat.op_end_rel_micros - node_stat.op_start_rel_micros
                elapsed_us_for_ops[name] = elapsed_us
        # Compute FLOPs of executed nodes.
        total_flops = 0
        for op in graph.get_operations():
            name = op.name
            if elapsed_us_for_ops.get(name, 0) > 0 and name in logged_ops:
                total_flops += logged_ops[name].float_ops
        return total_flops

# Note that we also need to declare how to compute FLOPs for each operation type present (that is, for
# each node type in the TensorFlow graph). We did this for the following operation types (and their
# gradients, where applicable):

#   • unary math operations: square, squre root, log, negation, element-wise inverse, softmax, L2 norm;
#   • binary element-wise operations: addition, subtraction, multiplication, division, minimum, maximum,
#       power, squared difference, comparison operations;
#   • reduction operations: mean, sum, argmax, argmin;
#   • convolution, average pooling, max pooling;
#   • matrix multiplication.
# For example, for the element-wise addition operation type:


@ops.RegisterStatistics("Add", "flops")
def _add_flops(graph, node):
    """Compute flops for the Add operation."""
    out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    out_shape.assert_is_fully_defined()
    return ops.OpStats("flops", out_shape.num_elements())
