import os

import dill
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from scipy.stats import spearmanr
from tqdm import tqdm

from xaibench.color import AVAIL_METHODS
from xaibench.score import distribute_bonds
from xaibench.utils import DATA_PATH, RESULTS_PATH


def method_agreement(color_all, metric_f):
    methods = list(color_all.keys())
    agreement = np.zeros((len(methods), len(methods)), dtype=np.float32)

    for idx_i, method_i in enumerate(methods):
        for idx_j, method_j in enumerate(methods):
            if idx_j > idx_i:
                color_i, color_j = color_all[method_i], color_all[method_j]
                m_i = [
                    metric_f(pair_i[0].flatten(), pair_j[0].flatten())
                    for pair_i, pair_j in zip(color_i, color_j)
                ]
                m_j = [
                    metric_f(pair_i[1].flatten(), pair_j[1].flatten())
                    for pair_i, pair_j in zip(color_i, color_j)
                ]
                agreement[idx_i, idx_j] = np.mean([m_i] + [m_j])

    agreement += agreement.T.copy()
    agreement[np.diag_indices_from(agreement)] = 1.0
    return agreement


if __name__ == "__main__":
    ids = os.listdir(os.path.join(DATA_PATH, "validation_sets"))

    ags = []

    for id_ in tqdm(ids):
        dirname = os.path.join(DATA_PATH, "validation_sets", id_)
        color_gcn = os.path.join(dirname, "colors_gcn.pt")
        color_mpnn = os.path.join(dirname, "colors_mpnn.pt")
        color_gat = os.path.join(dirname, "colors_gat.pt")
        color_graphnet = os.path.join(dirname, "colors_graphnet.pt")
        color_rf = os.path.join(dirname, "colors_rf.pt")
        color_dnn = os.path.join(dirname, "colors_dnn.pt")

        if all(
            [
                os.path.exists(color)
                for color in [
                    color_gcn,
                    color_mpnn,
                    color_gat,
                    color_graphnet,
                    color_rf,
                    color_dnn,
                ]
            ]
        ):
            pair_df = pd.read_csv(os.path.join(dirname, "pairs.csv"))
            mols = [
                (MolFromSmiles(mi), MolFromSmiles(mj))
                for mi, mj in zip(pair_df["smiles_i"], pair_df["smiles_j"])
            ]

            with open(color_gcn, "rb") as handle:
                color_gcn = dill.load(handle)

            with open(color_mpnn, "rb") as handle:
                color_mpnn = dill.load(handle)

            with open(color_gat, "rb") as handle:
                color_gat = dill.load(handle)

            with open(color_graphnet, "rb") as handle:
                color_graphnet = dill.load(handle)

            with open(color_rf, "rb") as handle:
                color_rf = dill.load(handle)

            with open(color_dnn, "rb") as handle:
                color_dnn = dill.load(handle)

            color_all = {}
            color_all["RF (Sheridan)"] = color_rf
            color_all["DNN (Sheridan)"] = color_dnn

            names = ["gcn", "mpnn", "gat", "graphnet"]

            # Distribute bond importance for gnn-based methods
            for method in AVAIL_METHODS + ["diff"]:
                if method != "diff":
                    method_name = method.__name__
                else:
                    method_name = method

                if method_name == "AttentionWeights":
                    gat_method = color_gat[method_name]

                    color_all["gat" + f" ({method_name})"] = [
                        (
                            distribute_bonds(cm_pair[0], mol_pair[0]),
                            distribute_bonds(cm_pair[1], mol_pair[1]),
                        )
                        for cm_pair, mol_pair in zip(gat_method, mols)
                    ]

                else:
                    gcn_method = color_gcn[method_name]
                    mpnn_method = color_mpnn[method_name]
                    gat_method = color_gat[method_name]
                    graphnet_method = color_graphnet[method_name]

                    if method_name != "diff":
                        for name, color_gnn in zip(
                            names,
                            [gcn_method, mpnn_method, gat_method, graphnet_method],
                        ):
                            color_all[name.upper() + f" ({method_name})"] = [
                                (
                                    distribute_bonds(cm_pair[0], mol_pair[0]),
                                    distribute_bonds(cm_pair[1], mol_pair[1]),
                                )
                                for cm_pair, mol_pair in zip(color_gnn, mols)
                            ]
                    else:
                        for name, color_gnn in zip(
                            names,
                            [gcn_method, mpnn_method, gat_method, graphnet_method],
                        ):
                            color_all[name.upper() + f" ({method_name})"] = color_gnn

            agreement = method_agreement(
                color_all, metric_f=lambda x, y: spearmanr(x, y).correlation
            )
            ags.append(agreement)

    
    with open(os.path.join(RESULTS_PATH, "method_agreement.pt"),"wb") as handle:
        dill.dump(ags, handle)

