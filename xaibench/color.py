import argparse
import os

import dill
import pandas as pd
from graph_attribution.featurization import MolTensorizer, smiles_to_graphs_tuple
from graph_attribution.graphnet_techniques import (
    CAM,
    AttentionWeights,
    GradCAM,
    GradInput,
    IntegratedGradients,
)
from graph_attribution.graphs import get_graphs_tf, get_num_graphs
from joblib import load as load_sklearn
from tqdm import tqdm

from xaibench.color_utils import get_batch_indices, ig_ref
from xaibench.diff_utils import diff_gnn, diff_rf
from xaibench.train_gnn import DEVICE
from xaibench.utils import DATA_PATH, MODELS_PATH, MODELS_RF_PATH

AVAIL_METHODS = [IntegratedGradients, GradInput, CAM, GradCAM, AttentionWeights]


def color_pairs(pair_df, model, batch_size=16, block_type="gcn"):
    """
    Uses the methods in AVAIL_METHODS alongside GNN models to color
    all pairs of molecules available in `pair_df`.
    """
    tensorizer = MolTensorizer()

    g_i, g_j = (
        smiles_to_graphs_tuple(pair_df["smiles_i"], tensorizer),
        smiles_to_graphs_tuple(pair_df["smiles_j"], tensorizer),
    )

    colors = {}

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

        for idx in tqdm(indices):
            with DEVICE:
                b_i, b_j = get_graphs_tf(g_i, idx), get_graphs_tf(g_j, idx)
                c_i = col_method(**extra_kwargs).attribute(b_i, model)
                c_j = col_method(**extra_kwargs).attribute(b_j, model)

            col_i.extend(c_i)
            col_j.extend(c_j)

        colors[col_method.__name__] = [(c_i, c_j) for c_i, c_j in zip(col_i, col_j)]
    return colors


def color_pairs_diff(pair_df, model, diff_fun):
    """
    Uses Sheridan's (2019) method to color all pairs of molecules
    available in `pair_df`.
    """
    colors = []

    for row in tqdm(pair_df.itertuples(), total=len(pair_df)):
        color_i, color_j = (
            diff_fun(getattr(row, "smiles_i"), model),
            diff_fun(getattr(row, "smiles_j"), model),
        )
        colors.append((color_i, color_j))
    return colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pairs", dest="pair_f", type=str, required=True,
    )
    parser.add_argument(
        "-block_type", dest="block_type", type=str, required=False, default="gcn"
    )
    args = parser.parse_args()

    ##########

    id_ = os.path.basename(os.path.dirname(args.pair_f))
    colors_pt = os.path.join(os.path.dirname(args.pair_f), "colors.pt")

    if not os.path.exists(colors_pt):
        print(f"No colors available for id {id_}. Skipping...")

    else:
        pair_df = pd.read_csv(args.pair_f)

        if args.block_type == "rf":
            model_rf = load_sklearn(os.path.join(MODELS_RF_PATH, f"{id_}.pt"))
            colors = color_pairs_diff(pair_df, model=model_rf, diff_fun=diff_rf)
        else:
            with open(
                os.path.join(MODELS_PATH, f"{args.block_type}_{id_}.pt"), "rb"
            ) as handle:
                model_gnn = dill.load(handle)

            colors = color_pairs(pair_df, model_gnn, block_type=args.block_type)
            colors["diff"] = color_pairs_diff(pair_df, model=model_gnn, diff_fun=diff_gnn)

        with open(
            os.path.join(DATA_PATH, "validation_sets", id_, f"colors_{args.block_type}.pt"),
            "wb",
        ) as handle:
            dill.dump(colors, handle)
