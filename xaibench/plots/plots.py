import collections
import os
from glob import glob

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.score import N_THRESHOLDS
from xaibench.utils import BLOCK_TYPES, DATA_PATH, FIG_PATH, RESULTS_PATH, LOG_PATH

if __name__ == "__main__":
    with open(os.path.join(RESULTS_PATH, "scores.pt"), "rb") as handle:
        scores = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "idxs.pt"), "rb") as handle:
        idxs = dill.load(handle)

    # median plot
    f, ax = plt.subplots(figsize=(8, 8))
    fontP = FontProperties()
    fontP.set_size("xx-small")
    cm = plt.get_cmap("tab20b")
    num_colors = (len(AVAIL_METHODS) + 2) * len(BLOCK_TYPES)
    ax.set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])

    ax.plot(
        MIN_PER_COMMON_ATOMS,
        [np.median(scores["rf"]["rf"][idx_th]) for idx_th in range(N_THRESHOLDS)],
        label="Sheridan",
        marker="o",
    )

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        for method in avail_methods:
            method_name = method if isinstance(method, str) else method.__name__
            medians = [
                np.median(scores[bt][method_name][idx_th])
                for idx_th in range(N_THRESHOLDS)
            ]
            ax.plot(
                MIN_PER_COMMON_ATOMS, medians, label=f"{bt}_{method_name}", marker="o"
            )
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

    colors_rf = np.array(scores["rf"])[idxs["rf"]["rf"][0]]

    for idx, color_f in enumerate(colors_rf):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
        if os.path.exists(sim_file):
            similarities_rf.append(np.load(sim_file).max())
            exists_rf.append(idx)

    y_rf = np.array(scores["rf"]["rf"][0])[exists_rf]

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 2 if bt == "gat" else len(AVAIL_METHODS) + 1

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))
        similarities = []
        exists_idx = []

        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        colors_bt = np.array(
            glob(os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",))
        )[idxs[bt]["CAM"][0]]

        for idx, color_f in enumerate(colors_bt):
            sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
            if os.path.exists(sim_file):
                similarities.append(np.load(sim_file).max())
                exists_idx.append(idx)

        axs[0].scatter(similarities_rf, y_rf, s=1.5)
        axs[0].set_title("Sheridan")
        axs[0].set_ylabel("Color agreement")
        axs[0].text(
            0.35, 0.9, "r={:.3f}".format(np.corrcoef(similarities_rf, y_rf)[0, 1])
        )

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y = np.array(scores[bt][method_name][0])[exists_idx]
            axs[idx_m + 1].scatter(similarities, y, s=1.5)
            axs[idx_m + 1].set_title(f"{method_name}")
            axs[idx_m + 1].text(
                0.35, 0.9, "r={:.3f}".format(np.corrcoef(similarities, y)[0, 1])
            )
        f.text(0.5, 0.02, "Training/test max. Tanimoto similarity", ha="center")
        plt.suptitle(f"Block type: {bt}")
        plt.savefig(os.path.join(FIG_PATH, f"sim_agreement_bond_{bt}.png"), dpi=300)
        plt.close()

    # training set size
    n_rf = []
    exists_rf = []

    for idx, color_f in enumerate(colors_rf):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(sim_file):
            n_rf.append(len(pd.read_csv(train_file)))
            exists_rf.append(idx)

    y_rf = np.array(scores["rf"]["rf"][0])[exists_rf]

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 2 if bt == "gat" else len(AVAIL_METHODS) + 1

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))
        n = []
        exists_idx = []

        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        colors_bt = np.array(
            glob(os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",))
        )[idxs[bt]["CAM"][0]]

        for idx, color_f in enumerate(colors_bt):
            train_file = os.path.join(os.path.dirname(color_f), "training.csv")
            if os.path.exists(train_file):
                n.append(len(pd.read_csv(train_file)))
                exists_idx.append(idx)

        axs[0].scatter(n_rf, y_rf, s=1.5)
        axs[0].set_title("Sheridan")
        axs[0].set_ylabel("Color agreement")
        axs[0].text(
            0.35, 0.9, "r={:.3f}".format(np.corrcoef(n_rf, y_rf)[0, 1])
        )

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y = np.array(scores[bt][method_name][0])[exists_idx]
            axs[idx_m + 1].scatter(n, y, s=1.5)
            axs[idx_m + 1].set_title(f"{method_name}")
            axs[idx_m + 1].text(
                0.35, 0.9, "r={:.3f}".format(np.corrcoef(n, y)[0, 1])
            )
        f.text(0.5, 0.02, "Number of training samples", ha="center")
        plt.suptitle(f"Block type: {bt}")
        plt.savefig(os.path.join(FIG_PATH, f"n_agreement_bond_{bt}.png"), dpi=300)
        plt.close()


    # performance

    all_metrics = {}
    all_metrics["rf"] = collections.defaultdict(list)
    exists_rf = []

    for idx, color_f in enumerate(colors_rf):
        id_ = os.path.basename(os.path.dirname(color_f))
        metrics_path = os.path.join(LOG_PATH, f"{id_}_metrics_rf.pt")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as handle:
                metrics = dill.load(handle)
                all_metrics["rf"]["rmse_train"].append(metrics["rmse_train"])
                all_metrics["rf"]["rmse_test"].append(metrics["rmse_test"])
                all_metrics["rf"]["pcc_train"].append(metrics["pcc_train"])
                all_metrics["rf"]["pcc_test"].append(metrics["pcc_test"])
            exists_rf.append(idx)

    y_rf = np.array(scores["rf"]["rf"][0])[exists_rf]

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 2 if bt == "gat" else len(AVAIL_METHODS) + 1
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))

        axs[0].scatter(losses_rf, y_rf, s=1.5)
        axs[0].set_title("Sheridan")
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

        all_metrics[bt] = collections.defaultdict(list)

        colors_bt = np.array(
            glob(os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",))
        )[idxs[bt]["CAM"][0]]

        for color_f in colors_bt:
            id_ = os.path.basename(os.path.dirname(color_f))

            with open(os.path.join(LOG_PATH, f"{bt}_{id_}.pt"), "rb") as handle:
                metrics = dill.load(handle)
            
            all_metrics[bt]["rmse_train"].append(metrics["rmse_train"][-1])
            all_metrics[bt]["rmse_test"].append(metrics["rmse_test"][-1])
            all_metrics[bt]["pcc_train"].append(metrics["pcc_train"][-1])
            all_metrics[bt]["pcc_test"].append(metrics["pcc_test"][-1])

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y = np.array(scores[bt][method_name][0])
            axs[idx_m + 1].scatter(losses, y, s=1.5)
            axs[idx_m + 1].set_title(f"{method_name}")
            axs[idx_m + 1].text(
                3.0,
                0.9,
                "r={:.3f}".format(
                    np.corrcoef(losses[~np.isnan(losses)], y[~np.isnan(losses)])[0, 1]
                ),
            )
        f.text(0.5, 0.02, "Train RMSE", ha="center")
        plt.suptitle(f"Block type: {bt}")
        plt.savefig(os.path.join(FIG_PATH, f"perf_agreement_bond_{bt}.png"), dpi=300)
        plt.close()
