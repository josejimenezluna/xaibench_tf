import os
import argparse
from glob import glob

import dill
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.utils import BLOCK_TYPES, DATA_PATH, FIG_PATH, RESULTS_PATH

N_THRESHOLDS = len(MIN_PER_COMMON_ATOMS)

f_score = lambda x, y: f1_score(x, y, zero_division=1)

def color_agreement(color_true, color_pred, metric_f):
    """
    Checks agreement between true and predicted colors.
    """
    assert len(color_true) == len(color_pred)
    idx_noncommon = [idx for idx, val in color_true.items() if val != 0.0]
    if len(idx_noncommon) == 0:
        return -1.0
    color_true_noncommon = np.array([color_true[idx] for idx in idx_noncommon])
    color_pred_noncommon = np.sign([color_pred[idx] for idx in idx_noncommon])

    # Check that no zeros exists after sign function
    color_true_noncommon = color_true_noncommon[color_pred_noncommon != 0]
    color_pred_noncommon = color_pred_noncommon[color_pred_noncommon != 0]
    if len(color_true_noncommon) == 0:
        return -1.0
    return metric_f(color_true_noncommon, color_pred_noncommon)


def distribute_bonds(cm, mol):
    """
    Distributes bond importances evenly across their connecting nodes.
    """
    atom_imp = cm.nodes.numpy()
    bond_imp = np.array([b for idx, b in enumerate(cm.edges.numpy()) if idx % 2 == 0])

    for bond_idx, bond in enumerate(mol.GetBonds()):
        b_imp = bond_imp[bond_idx] / 2
        atom_imp[bond.GetBeginAtomIdx()] += b_imp
        atom_imp[bond.GetEndAtomIdx()] += b_imp
    return atom_imp


def method_comparison(
    colors_path, other_name=None, avail_methods=None, assign_bonds=False
):
    avg_acc = {}
    avg_f1 = {}
    idx_valid = {}

    for idx, color_method_f in enumerate(tqdm(colors_path)):
        dirname = os.path.dirname(color_method_f)

        with open(os.path.join(dirname, "colors.pt"), "rb") as handle:
            colors = dill.load(handle)

        with open(color_method_f, "rb") as handle:
            manual_colors = dill.load(handle)

        # If bond importances are used, bond information from the mols is needed
        if assign_bonds:
            pair_df = pd.read_csv(os.path.join(dirname, "pairs.csv"))
            mols = [
                (MolFromSmiles(mi), MolFromSmiles(mj))
                for mi, mj in zip(pair_df["smiles_i"], pair_df["smiles_j"])
            ]
            assert len(mols) == len(colors)

        for method in avail_methods if avail_methods is not None else [other_name]:
            if avail_methods is not None:
                if method == "diff":
                    method_name = method
                else:
                    method_name = method.__name__

                if assign_bonds and method_name != "diff":
                    colors_method = [
                        (
                            distribute_bonds(cm_pair[0], mol_pair[0]),
                            distribute_bonds(cm_pair[1], mol_pair[1]),
                        )
                        for cm_pair, mol_pair in zip(manual_colors[method_name], mols)
                    ]
                elif method_name != "diff":
                    colors_method = [
                        (cm[0].nodes.numpy(), cm[1].nodes.numpy())
                        for cm in manual_colors[method_name]
                    ]
                else:
                    colors_method = manual_colors[method_name]

            else:
                colors_method = manual_colors
                method_name = method

            assert len(colors) == len(colors_method)

            for idx_th in range(N_THRESHOLDS):
                colors_th = [col[idx_th] for col in colors]

                if sum(1 for _ in filter(None.__ne__, colors_th)) > 0:
                    accs = []
                    f1s = []

                    for color_pair_true, color_pair_pred in zip(
                        colors_th, colors_method
                    ):
                        if color_pair_true is not None:
                            acc_i = color_agreement(
                                color_pair_true[0],
                                color_pair_pred[0],
                                metric_f=accuracy_score,
                            )
                            accs.append(acc_i)
                            acc_j = color_agreement(
                                color_pair_true[1],
                                color_pair_pred[1],
                                metric_f=accuracy_score,
                            )
                            accs.append(acc_j)

                            f1_i = color_agreement(
                                color_pair_true[0],
                                color_pair_pred[0],
                                metric_f=f_score,
                            )
                            f1s.append(f1_i)

                            f1_j = color_agreement(
                                color_pair_true[1],
                                color_pair_pred[1],
                                metric_f=f_score,
                            )
                            f1s.append(f1_j)

                    accs = np.array(accs)
                    accs = accs[
                        accs >= 0.0
                    ]  # Filter examples with non-common MCS

                    f1s = np.array(f1s)
                    f1s = f1s[
                        f1s >= 0.0
                    ]

                    if accs.size > 0:
                        avg_acc.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(accs.mean())
                        avg_f1.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(f1s.mean())
                        idx_valid.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(idx)
    return avg_acc, avg_f1, idx_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-savename", dest="savename", type=str, required=False, default=""
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(FIG_PATH, exist_ok=True)

    colors_rf = sorted(
        glob(
            os.path.join(
                DATA_PATH, "validation_sets", "*", f"colors_rf{args.savename}.pt"
            )
        )
    )
    colors_dnn = sorted(
        glob(
            os.path.join(
                DATA_PATH, "validation_sets", "*", f"colors_dnn{args.savename}.pt"
            )
        )
    )

    accs = {}
    f1s = {}
    idxs = {}

    accs["rf"] = {}
    f1s["rf"] = {}
    idxs["rf"] = {}

    accs["dnn"] = {}
    f1s["dnn"] = {}
    idxs["dnn"] = {}

    accs["rf"], f1s["rf"], idxs["rf"] = method_comparison(colors_rf, other_name="rf")
    accs["dnn"], f1s["dnn"], idxs["dnn"] = method_comparison(colors_dnn, other_name="dnn")

    for bt in BLOCK_TYPES:
        print(f"Now loading block type {bt}...")
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]  # TODO: rewrite this more elegantly

        colors_method = sorted(
            glob(
                os.path.join(
                    DATA_PATH, "validation_sets", "*", f"colors_{bt}{args.savename}.pt",
                )
            )
        )

        accs[bt], f1s[bt], idxs[bt] = method_comparison(
            colors_method, avail_methods=avail_methods, assign_bonds=True
        )

    with open(os.path.join(RESULTS_PATH, f"accs{args.savename}.pt"), "wb") as handle:
        dill.dump(accs, handle)

    with open(os.path.join(RESULTS_PATH, f"f1s{args.savename}.pt"), "wb") as handle:
        dill.dump(f1s, handle)

    with open(os.path.join(RESULTS_PATH, f"idxs{args.savename}.pt"), "wb") as handle:
        dill.dump(idxs, handle)
