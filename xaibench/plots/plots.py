import collections
import os
from glob import glob

import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.score import N_THRESHOLDS
from xaibench.utils import BLOCK_TYPES, DATA_PATH, FIG_PATH, LOG_PATH, RESULTS_PATH

matplotlib.use("Agg")


def comparison_plot(xs, ys, block_type, avail_methods, common_x_label, savename):
    ncols = len(avail_methods) + 1
    f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))
    axs[0].scatter(xs["rf"], ys["rf"], s=1.5)
    axs[0].set_title("Sheridan")
    axs[0].set_ylabel("Color agreement")
    axs[0].text(
        0.1,
        0.9,
        "r={:.3f}".format(np.corrcoef(xs["rf"], ys["rf"])[0, 1]),
        va="center",
        ha="center",
        transform=axs[0].transAxes,
    )

    for idx_m, method in enumerate(avail_methods):
        method_name = method if isinstance(method, str) else method.__name__

        axs[idx_m + 1].scatter(xs[block_type], ys[block_type][method_name], s=1.5)
        axs[idx_m + 1].set_title(f"{method_name}")
        axs[idx_m + 1].text(
            0.25,
            0.9,
            "r={:.3f}".format(
                np.corrcoef(xs[block_type], ys[block_type][method_name])[0, 1]
            ),
            ha="center",
            va="center",
            transform=axs[idx_m + 1].transAxes,
        )

    f.text(0.5, 0.02, common_x_label, ha="center")
    plt.suptitle(f"Block type: {block_type}")
    plt.savefig(os.path.join(FIG_PATH, f"{savename}_{block_type}.png"), dpi=300)
    plt.close()


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

    ## Common variables for other plots

    colors = {}
    colors["rf"] = np.array(
        sorted(glob(os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf.pt")))
    )[
        idxs["rf"]["rf"][0]
    ]  # 0 is at MCS threshold .5

    for bt in BLOCK_TYPES:
        colors[bt] = np.array(
            sorted(
                glob(
                    os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}.pt",)
                )
            )
        )[idxs[bt]["CAM"][0]]

    # similarities
    similarities = collections.defaultdict(list)
    exists = collections.defaultdict(list)

    for idx, color_f in enumerate(tqdm(colors["rf"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
        if os.path.exists(sim_file):
            similarities["rf"].append(np.load(sim_file).mean())
            exists["rf"].append(idx)

    y = {}
    y["rf"] = np.array(scores["rf"]["rf"][0])[exists["rf"]]

    for bt in BLOCK_TYPES:
        ncols = len(AVAIL_METHODS) + 2 if bt == "gat" else len(AVAIL_METHODS) + 1

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))

        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors[bt])):
            sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
            if os.path.exists(sim_file):
                similarities[bt].append(np.load(sim_file).mean())
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(scores[bt][method_name][0])[exists[bt]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            similarities,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train/test mean Tanimoto similarity",
            savename="similarity",
        )

    # training set size
    sizes = collections.defaultdict(list)
    exists = collections.defaultdict(list)

    for idx, color_f in enumerate(colors["rf"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(sim_file):
            sizes["rf"].append(len(pd.read_csv(train_file)))
            exists["rf"].append(idx)

    y = {}
    y["rf"] = np.array(scores["rf"]["rf"][0])[exists["rf"]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors[bt])):
            train_file = os.path.join(os.path.dirname(color_f), "training.csv")
            if os.path.exists(train_file):
                sizes[bt].append(len(pd.read_csv(train_file)))
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(tqdm(avail_methods)):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(scores[bt][method_name][0])[exists[bt]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            sizes,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Number of training samples",
            savename="sizes",
        )

    # performance
    all_metrics = {}
    all_metrics["rmse_train"] = collections.defaultdict(list)
    all_metrics["rmse_test"] = collections.defaultdict(list)
    all_metrics["pcc_train"] = collections.defaultdict(list)
    all_metrics["pcc_test"] = collections.defaultdict(list)

    exists = collections.defaultdict(list)

    for idx, color_f in enumerate(tqdm(colors["rf"])):
        id_ = os.path.basename(os.path.dirname(color_f))
        metrics_path = os.path.join(LOG_PATH, f"{id_}_metrics_rf.pt")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as handle:
                metrics = dill.load(handle)
                all_metrics["rmse_train"]["rf"].append(metrics["rmse_train"])
                all_metrics["rmse_test"]["rf"].append(metrics["rmse_test"])
                all_metrics["pcc_train"]["rf"].append(metrics["pcc_train"])
                all_metrics["pcc_test"]["rf"].append(metrics["pcc_test"])
            exists["rf"].append(idx)

    y = {}
    y["rf"] = np.array(scores["rf"]["rf"][0])[exists["rf"]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))

        for color_f in tqdm(colors[bt]):
            id_ = os.path.basename(os.path.dirname(color_f))

            with open(os.path.join(LOG_PATH, f"{bt}_{id_}.pt"), "rb") as handle:
                metrics = dill.load(handle)

            all_metrics["rmse_train"][bt].append(metrics["rmse_train"][-1])
            all_metrics["rmse_test"][bt].append(metrics["rmse_test"][-1])
            all_metrics["pcc_train"][bt].append(metrics["pcc_train"][-1])
            all_metrics["pcc_test"][bt].append(metrics["pcc_test"][-1])

        y[bt] = {}

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(scores[bt][method_name][0])

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        comparison_plot(
            all_metrics["rmse_train"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train RMSE",
            savename="train_rmse",
        )
        comparison_plot(
            all_metrics["rmse_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test RMSE",
            savename="test_rmse",
        )
        comparison_plot(
            all_metrics["pcc_train"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train PCC",
            savename="train_pcc",
        )
        comparison_plot(
            all_metrics["pcc_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test PCC",
            savename="test_pcc",
        )

