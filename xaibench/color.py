import argparse
import os

import dill
import pandas as pd
import tensorflow as tf
from graph_attribution.featurization import MolTensorizer, smiles_to_graphs_tuple
from graph_attribution.graphnet_techniques import (
    IntegratedGradients,
    CAM,
    GradCAM,
    GradInput,
    AttentionWeights,
)
from graph_nets.graphs import GraphsTuple
from graph_attribution.graphs import get_num_graphs, get_graphs_tf

from xaibench.utils import DATA_PATH, MODELS_PATH

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

AVAIL_METHODS = [IntegratedGradients, GradInput, CAM, GradCAM, AttentionWeights]


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
    n_batches = n // batch_size
    indices = indices[: n_batches * batch_size]
    indices = tf.reshape(indices, (n_batches, batch_size))
    return indices


def color_pairs(pair_f, batch_size=16, block_type="gcn"):
    id_ = os.path.basename(os.path.dirname(pair_f))
    colors_pt = os.path.join(os.path.dirname(pair_f), "colors.pt")

    if not os.path.exists(colors_pt):
        raise ValueError(f"No colors available for id {id_}. Skipping...")

    df = pd.read_csv(pair_f)
    tensorizer = MolTensorizer()

    g_i, g_j = (
        smiles_to_graphs_tuple(df["smiles_i"], tensorizer),
        smiles_to_graphs_tuple(df["smiles_j"], tensorizer),
    )

    colors = {}

    with open(os.path.join(MODELS_PATH, f"{block_type}_{id_}.pt"), "rb") as handle:
        model = dill.load(handle)

    for col_method in AVAIL_METHODS:
        extra_kwargs = {}
        col_i = []
        col_j = []

        if col_method == AttentionWeights and block_type != "gat":
            continue

        if col_method == IntegratedGradients:
            extra_kwargs["num_steps"] = 500
            extra_kwargs["reference_fn"] = ig_ref

        n = get_num_graphs(g_i)
        indices = get_batch_indices(n, int(batch_size / 2))

        for idx in indices:
            with tf.device("/GPU:0"):
                b_i, b_j = get_graphs_tf(g_i, idx), get_graphs_tf(g_j, idx)
                c_i = col_method(**extra_kwargs).attribute(b_i, model)
                c_j = col_method(**extra_kwargs).attribute(b_j, model)

            col_i.extend(c_i)
            col_j.extend(c_j)

        colors[col_method.__name__] = [(c_i, c_j) for c_i, c_j in zip(col_i, col_j)]
    return colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pairs", dest="pair_f", type=str, required=True,
    )
    parser.add_argument(
        "-bt", dest="block_type", type=str, required=False, default="gcn"
    )
    args = parser.parse_args()

    colors = color_pairs(pair_f=args.pair_f, block_type=args.block_type)
    id_ = os.path.basename(os.path.dirname(args.pair_f))

    with open(
        os.path.join(DATA_PATH, "validation_sets", id_, f"colors_{args.block_type}.pt"),
        "wb",
    ) as handle:
        dill.dump(colors, handle)
