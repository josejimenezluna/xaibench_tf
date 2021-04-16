import tensorflow as tf
from graph_nets.graphs import GraphsTuple


def ig_ref(g):
    nodes = g.nodes * 0.0
    edges = g.edges * 0.0
    g_ref = GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=g.receivers,
        senders=g.senders,
        globals=g.globals,
        n_node=g.n_node,
        n_edge=g.n_edge,
    )
    return g_ref


def get_batch_indices(n: int, batch_size: int):
    indices = tf.range(n)
    if n < batch_size:
        indices = tf.reshape(indices, (1, n))
        return indices
    if n % batch_size == 0:
        n_batches = n // batch_size
    else:
        n_batches = n // batch_size + 1
    return [
        indices[idx * batch_size : batch_size * (idx + 1)] for idx in range(n_batches)
    ]
