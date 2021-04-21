import os
from glob import glob

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.utils import BLOCK_TYPES, DATA_PATH, FIG_PATH, LOG_PATH, RESULTS_PATH

N_THRESHOLDS = len(MIN_PER_COMMON_ATOMS)


def color_agreement(color_true, color_pred, metric_f):
    assert len(color_true) == len(color_pred)
    idx_noncommon = [idx for idx, val in color_true.items() if val != 0.0]
    if len(idx_noncommon) == 0:
        return -1.0
    color_true_noncommon = np.array([color_true[idx] for idx in idx_noncommon])
    color_pred_noncommon = np.sign([color_pred[idx] for idx in idx_noncommon])
    return metric_f(color_true_noncommon, color_pred_noncommon)


def distribute_bonds(cm, mol):
    atom_imp = cm.nodes.numpy()
    bond_imp = np.array([b for idx, b in enumerate(cm.edges.numpy()) if idx % 2 == 0])

    for bond_idx, bond in enumerate(mol.GetBonds()):
        b_imp = bond_imp[bond_idx] / 2
        atom_imp[bond.GetBeginAtomIdx()] += b_imp
        atom_imp[bond.GetEndAtomIdx()] += b_imp
    return atom_imp


#TODO: this function needs to be refactored
def method_comparison(colors_path, avail_methods=None, assign_bonds=False):
    avg_scores = {}
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

        for method in avail_methods if avail_methods is not None else ["rf"]:
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

            for idx_th in range(N_THRESHOLDS):
                colors_th = [col[idx_th] for col in colors]

                assert len(colors) == len(colors_method)

                if sum(1 for _ in filter(None.__ne__, colors)) > 0:
                    scores = []
                    for color_pair_true, color_pair_pred in zip(
                        colors_th, colors_method
                    ):
                        if color_pair_true is not None:
                            ag_i = color_agreement(
                                color_pair_true[0],
                                color_pair_pred[0],
                                metric_f=accuracy_score,
                            )
                            scores.append(ag_i)
                            ag_j = color_agreement(
                                color_pair_true[1],
                                color_pair_pred[1],
                                metric_f=accuracy_score,
                            )
                            scores.append(ag_j)

                    scores = np.array(scores)
                    scores = scores[
                        scores >= 0.0
                    ]  # Filter examples with non-common MCS

                    if scores.size > 0:
                        avg_scores.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(scores.mean())
                        idx_valid.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(idx)
    return avg_scores, idx_valid


if __name__ == "__main__":
    os.makedirs(RESULTS_PATH, exist_ok=True)
    results_path = os.path.join(RESULTS_PATH, "scores.pt")

    if not os.path.exists(results_path):
        colors_rf = glob(
            os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf.pt")
        )

        scores = {}
        idxs = {}
        scores["rf"] = {}
        idxs["rf"] = {}

        scores["rf"], idxs["rf"] = method_comparison(colors_rf)

        for bt in BLOCK_TYPES:
            print(f"Now loading block type {bt}...")
            avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
            avail_methods = avail_methods + [
                "diff"
            ]  # TODO: rewrite this more elegantly

            colors_method = glob(
                os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",)
            )

            scores[bt], idxs[bt] = method_comparison(
                colors_method, avail_methods, assign_bonds=True
            )

        with open(results_path, "wb") as handle:
            dill.dump(scores, handle)

    else:
        with open(results_path, "rb") as handle:
            scores = dill.load(handle)

    # Histograms per idx_th
    os.makedirs(FIG_PATH, exist_ok=True)

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 1 if bt == "gat" else len(AVAIL_METHODS)

        avail_methods = (
            AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        )  # TODO: rewrite this more elegantly
        for idx_th in range(N_THRESHOLDS):
            f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 6))
            axs[0].hist(scores["rf"]["rf"][idx_th], bins=50)
            axs[0].axvline(
                np.median(scores["rf"]["rf"][idx_th]), linestyle="--", color="black"
            )
            axs[0].set_xlabel("Diff.")
            for idx_method, method in enumerate(avail_methods):
                s = scores[bt][method.__name__][idx_th]
                axs[idx_method + 1].hist(s, bins=50)
                axs[idx_method + 1].axvline(np.median(s), linestyle="--", color="black")
                axs[idx_method + 1].set_xlabel(method.__name__)

            plt.suptitle(
                f"Average agreement between attributions and coloring \n Block type: {bt} (bond), MCS Threshold: {MIN_PER_COMMON_ATOMS[idx_th]:.2f}"
            )
            plt.savefig(
                os.path.join(FIG_PATH, f"color_agreement_bond_{bt}_{idx_th}.png"),
                dpi=300,
            )
            plt.close()

    # median plot
    f, ax = plt.subplots(figsize=(8, 8))
    fontP = FontProperties()
    fontP.set_size("xx-small")

    ax.plot(
        MIN_PER_COMMON_ATOMS,
        [np.median(scores["rf"]["rf"][idx_th]) for idx_th in range(N_THRESHOLDS)],
        label="Diff.",
    )

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        for method in avail_methods:
            medians = [
                np.median(scores[bt][method.__name__][idx_th])
                for idx_th in range(N_THRESHOLDS)
            ]
            ax.plot(MIN_PER_COMMON_ATOMS, medians, label=f"{bt}_{method.__name__}")
        ax.grid(True)
    ax.set_xlabel("MCS percentage common atoms (0-1)")
    ax.set_ylabel("Color agreement")
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop=fontP,
        fancybox=True,
        shadow=True,
    )
    plt.subplots_adjust(right=0.75)
    plt.savefig(
        os.path.join(FIG_PATH, f"color_agreement_medians_bond.png"), dpi=300,
    )
    plt.close()

    # similarities
    similarities_rf = []
    exists_rf = []

    colors_rf = np.array(colors_rf)[idxs["rf"]["rf"][0]]

    for idx, color_f in enumerate(colors_rf):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
        if os.path.exists(sim_file):
            similarities_rf.append(np.load(sim_file).max())
            exists_rf.append(idx)

    y_rf = np.array(scores["rf"]["rf"][0])[exists_rf]

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 1 if bt == "gat" else len(AVAIL_METHODS)

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 4))
        similarities = []
        exists_idx = []

        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]

        colors_bt = np.array(
            glob(os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",))
        )[idxs[bt]["CAM"][0]]

        for idx, color_f in enumerate(colors_bt):
            sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
            if os.path.exists(sim_file):
                similarities.append(np.load(sim_file).max())
                exists_idx.append(idx)

        axs[0].scatter(similarities_rf, y_rf, s=1.5)
        axs[0].set_title("Diff.")
        axs[0].set_ylabel("Color agreement")
        axs[0].text(
            0.35, 0.9, "r={:.3f}".format(np.corrcoef(similarities_rf, y_rf)[0, 1])
        )

        for idx_m, method in enumerate(avail_methods):
            y = np.array(scores[bt][method.__name__][0])[exists_idx]
            axs[idx_m + 1].scatter(similarities, y, s=1.5)
            axs[idx_m + 1].set_title(f"{method.__name__}")
            axs[idx_m + 1].text(
                0.35, 0.9, "r={:.3f}".format(np.corrcoef(similarities, y)[0, 1])
            )
        f.text(0.5, 0.04, "Training/test max. Tanimoto similarity", ha="center")
        plt.suptitle(f"Block type: {bt}")
        plt.savefig(os.path.join(FIG_PATH, f"sim_agreement_bond_{bt}.png"), dpi=300)
        plt.close()

    # performance
    losses_rf = []
    exists_rf = []

    for idx, color_f in enumerate(colors_rf):
        id_ = os.path.basename(os.path.dirname(color_f))
        metrics_path = os.path.join(LOG_PATH, f"{id_}_metrics_rf.pt")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as handle:
                losses_rf.append(dill.load(handle)[0])
            exists_rf.append(idx)

    losses_rf = np.array(losses_rf)
    y_rf = np.array(scores["rf"]["rf"][0])[exists_rf]

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 1 if bt == "gat" else len(AVAIL_METHODS)
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 4))

        axs[0].scatter(losses_rf, y_rf, s=1.5)
        axs[0].set_title("Diff.")
        axs[0].set_ylabel("Color agreement")
        axs[0].text(
            1.0,
            0.9,
            "r={:.3f}".format(
                np.corrcoef(
                    losses_rf[~np.isnan(losses_rf)], y_rf[~np.isnan(losses_rf)]
                )[0, 1]
            ),
        )

        losses = []

        colors_bt = np.array(
            glob(os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",))
        )[idxs[bt]["CAM"][0]]

        for color_f in colors_bt:
            id_ = os.path.basename(os.path.dirname(color_f))

            with open(os.path.join(LOG_PATH, f"{bt}_{id_}.pt"), "rb") as handle:
                losses.append(dill.load(handle)["train"][-1])

        losses = np.array(losses)
        for idx_m, method in enumerate(avail_methods):
            y = np.array(scores[bt][method.__name__][0])
            axs[idx_m + 1].scatter(losses, y, s=1.5)
            axs[idx_m + 1].set_title(f"{method.__name__}")
            axs[idx_m + 1].text(
                4.0,
                0.9,
                "r={:.3f}".format(
                    np.corrcoef(losses[~np.isnan(losses)], y[~np.isnan(losses)])[0, 1]
                ),
            )
        f.text(0.5, 0.04, "Train MSE (bond)", ha="center")
        plt.suptitle(f"Block type: {bt}")
        plt.savefig(os.path.join(FIG_PATH, f"perf_agreement_bond_{bt}.png"), dpi=300)
        plt.close()
