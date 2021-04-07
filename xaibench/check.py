import os
from glob import glob

import dill
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles

from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.utils import DATA_PATH, FIG_PATH, BLOCK_TYPES

N_THRESHOLDS = len(MIN_PER_COMMON_ATOMS)


def color_agreement(color_true, color_pred):
    assert len(color_true) == len(color_pred)
    idx_noncommon = [idx for idx, val in color_true.items() if val != 0.0]
    if len(idx_noncommon) == 0:
        return -1.0
    color_true_noncommon = np.array([color_true[idx] for idx in idx_noncommon])
    color_pred_noncommon = np.sign([color_pred[idx] for idx in idx_noncommon])
    agreement = color_true_noncommon == color_pred_noncommon
    return np.mean(agreement)


def assign_bonds(cm, mol):
    atom_imp = cm.nodes.numpy()
    bond_imp = [cm.edges[idx].numpy() for idx in range(len(cm.edges)) if idx % 2 == 0]

    for bond_idx, bond in enumerate(mol.GetBonds()):
        b_imp = bond_imp[bond_idx] / 2
        atom_imp[bond.GetBeginAtomIdx()] += b_imp
        atom_imp[bond.GetEndAtomIdx()] += b_imp
    return atom_imp


def method_comparison(colors_path, avail_methods=None, assign_bonds=False):
    avg_scores = {}
    idx_valid = {}

    for idx, color_method_f in enumerate(tqdm(colors_path)):
        dirname = os.path.dirname(color_method_f)

        with open(os.path.join(dirname, "colors.pt"), "rb") as handle:
            colors = dill.load(handle)

        if assign_bonds:
            pair_df = pd.read_csv(os.path.join(dirname, "pairs.csv"))
            mols = [
                (MolFromSmiles(mi), MolFromSmiles(mj))
                for mi, mj in zip(pair_df["smiles_i"], pair_df["smiles_j"])
            ]

        with open(color_method_f, "rb") as handle:
            manual_colors = dill.load(handle)

        for method in avail_methods if avail_methods is not None else ["rf"]:
            if avail_methods is not None:
                method_name = method.__name__
                if assign_bonds:
                    colors_method = [
                        (assign_bonds(cm[0], mol[0]), assign_bonds(cm[1], mol[1]))
                        for cm, mol in zip(manual_colors[method.__name__], mols)
                    ]
                else:
                    colors_method = [
                        (cm[0].nodes.numpy(), cm[1].nodes.numpy())
                        for cm in manual_colors[method.__name__]
                    ]
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
                                color_pair_true[0], color_pair_pred[0]
                            )
                            scores.append(ag_i)
                            ag_j = color_agreement(
                                color_pair_true[1], color_pair_pred[1]
                            )
                            scores.append(ag_j)

                    scores = np.array(scores)
                    scores = scores[scores >= 0.0]

                    if scores.size > 0:
                        avg_scores.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(scores.mean())
                        idx_valid.setdefault(
                            method_name, [[] for _ in range(N_THRESHOLDS)]
                        )[idx_th].append(idx)
    return avg_scores, idx_valid


if __name__ == "__main__":
    # Precompute results
    colors_rf = glob(os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf.pt"))

    scores = {}
    scores["rf"] = {}

    scores["rf"], _ = method_comparison(colors_rf)

    for bt in BLOCK_TYPES:
        print(f"Now loading block type {bt}...")
        avail_methods = (
            AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        )  # TODO: rewrite this more elegantly
        colors_method = glob(
            os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",)
        )

        scores[bt], _ = method_comparison(
            colors_method, avail_methods, assign_bonds=False
        )

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
                f"Average agreement between attributions and coloring \n Block type: {bt} (no bond), MCS Threshold: {MIN_PER_COMMON_ATOMS[idx_th]:.2f}"
            )
            plt.savefig(
                os.path.join(FIG_PATH, f"color_agreement_{bt}_{idx_th}.png"), dpi=300,
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
    plt.subplots_adjust(right=0.7)
    plt.show()
    # plt.close()

    # TODO: these plots need to be redone

    # similarities
    # exist_idx_sim_molgrad = []
    # similarities_molgrad = []

    # for idx, c in enumerate(colors_molgrad_all):
    #     sim_file = os.path.join(os.path.dirname(c), "similarity.npy")
    #     if os.path.exists(sim_file):
    #         similarities_molgrad.append(sim_file)
    #         exist_idx_sim_molgrad.append(idx)

    # exist_idx_sim_rf = []
    # similarities_rf = []

    # for idx, c in enumerate(colors_rf_all):
    #     sim_file = os.path.join(os.path.dirname(c), "similarity.npy")
    #     if os.path.exists(sim_file):
    #         similarities_rf.append(sim_file)
    #         exist_idx_sim_rf.append(idx)

    # avg_sim_molgrad = []
    # max_sim_molgrad = []

    # for sim_fs in similarities_molgrad:
    #     sim = np.load(sim_fs)
    #     avg_sim_molgrad.append(sim.mean())
    #     max_sim_molgrad.append(sim.max())

    # avg_sim_rf = []
    # max_sim_rf = []

    # for sim_fs in similarities_rf:
    #     sim = np.load(sim_fs)
    #     avg_sim_rf.append(sim.mean())
    #     max_sim_rf.append(sim.max())

    # f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axs[0].scatter(avg_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad], s=1.5)
    # axs[1].scatter(avg_sim_rf, scores_rf[exist_idx_sim_rf], s=1.5)
    # axs[0].set_ylabel("Agreement between attributions and coloring")
    # axs[0].set_xlabel("Avg. Tanimoto similarities between train and test sets")
    # axs[0].set_title("IG")
    # axs[0].text(
    #     0.3,
    #     0.9,
    #     "r={:.3f}".format(
    #         np.corrcoef(avg_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad])[0, 1]
    #     ),
    #     style="italic",
    #     bbox={"facecolor": "red", "alpha": 0.5},
    # )
    # axs[1].set_ylabel("Agreement between attributions and coloring")
    # axs[1].set_xlabel("Avg. Tanimoto similarities between train and test sets")
    # axs[1].set_title("Sheridan")
    # axs[1].text(
    #     0.3,
    #     0.9,
    #     "r={:.3f}".format(np.corrcoef(avg_sim_rf, scores_rf[exist_idx_sim_rf])[0, 1]),
    #     style="italic",
    #     bbox={"facecolor": "red", "alpha": 0.5},
    # )
    # plt.savefig(os.path.join(FIG_PATH, "avgsimilarityvsagreement.png"))
    # plt.close()

    # # max
    # f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axs[0].scatter(max_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad], s=1.5)
    # axs[1].scatter(max_sim_rf, scores_rf[exist_idx_sim_rf], s=1.5)
    # axs[0].set_ylabel("Agreement between attributions and coloring")
    # axs[0].set_xlabel("Max. Tanimoto similarities between train and test sets")
    # axs[0].set_title("IG")
    # axs[0].text(
    #     0.3,
    #     0.9,
    #     "r={:.3f}".format(
    #         np.corrcoef(max_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad])[0, 1]
    #     ),
    #     style="italic",
    #     bbox={"facecolor": "red", "alpha": 0.5},
    # )
    # axs[1].set_ylabel("Agreement between attributions and coloring")
    # axs[1].set_xlabel("Max. Tanimoto similarities between train and test sets")
    # axs[1].set_title("Sheridan")
    # axs[1].text(
    #     0.3,
    #     0.9,
    #     "r={:.3f}".format(np.corrcoef(max_sim_rf, scores_rf[exist_idx_sim_rf])[0, 1]),
    #     style="italic",
    #     bbox={"facecolor": "red", "alpha": 0.5},
    # )
    # plt.savefig(os.path.join(FIG_PATH, "maxsimilarityvsagreement.png"))
    # plt.close()

    # # performance
    # exist_idx_log_molgrad = []
    # rs_molgrad = []

    # for idx, c in enumerate(colors_molgrad_all):
    #     log_file = os.path.join(
    #         LOG_PATH, f"{os.path.basename(os.path.dirname(c))}_metrics.pt"
    #     )
    #     if os.path.exists(log_file):
    #         with open(log_file, "rb") as handle:
    #             metrics = pickle.load(handle)
    #             r = metrics[1][0]
    #         rs_molgrad.append(r)
    #         exist_idx_log_molgrad.append(idx)

    # exist_idx_log_rf = []
    # rs_rf = []

    # for idx, c in enumerate(colors_molgrad_all):
    #     log_file = os.path.join(
    #         LOG_PATH, f"{os.path.basename(os.path.dirname(c))}_metrics_rf.pt"
    #     )
    #     if os.path.exists(log_file):
    #         with open(log_file, "rb") as handle:
    #             metrics = pickle.load(handle)
    #             r = metrics[1]
    #         rs_rf.append(r)
    #         exist_idx_log_rf.append(idx)

    # f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axs[0].scatter(rs_molgrad, scores_molgrad[exist_idx_log_molgrad], s=1.5)
    # axs[1].scatter(rs_rf, scores_rf[exist_idx_log_rf], s=1.5)
    # axs[0].set_ylabel("Agreement between attributions and coloring")
    # axs[0].set_xlabel("Correlation on held-out test set")
    # axs[0].set_title("IG")
    # axs[1].set_ylabel("Agreement between attributions and coloring")
    # axs[1].set_xlabel("Correlation on held-out test set")
    # axs[1].set_title("Sheridan")
    # plt.savefig(os.path.join(FIG_PATH, "performancevsagreement.png"))
    # plt.close()
