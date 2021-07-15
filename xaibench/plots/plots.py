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

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)


def comparison_plot(xs, ys, block_type, avail_methods, common_x_label, savename):
    ncols = len(avail_methods) + 2  # +2 for sheridan rf, dnn
    f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))
    axs[0].scatter(xs["rf"], ys["rf"], s=1.5)
    axs[0].set_title("Sheridan (RF)")
    axs[0].set_ylabel("Color accuracy")
    axs[0].text(
        0.25,
        0.9,
        "r={:.3f}".format(np.corrcoef(xs["rf"], ys["rf"])[0, 1]),
        va="center",
        ha="center",
        transform=axs[0].transAxes,
    )

    axs[1].scatter(xs["dnn"], ys["dnn"], s=1.5)
    axs[1].set_title("Sheridan (DNN)")
    axs[1].text(
        0.25,
        0.9,
        "r={:.3f}".format(np.corrcoef(xs["dnn"], ys["dnn"])[0, 1]),
        va="center",
        ha="center",
        transform=axs[1].transAxes,
    )

    for idx_m, method in enumerate(avail_methods):
        method_name = method if isinstance(method, str) else method.__name__

        axs[idx_m + 2].scatter(xs[block_type], ys[block_type][method_name], s=1.5)
        axs[idx_m + 2].set_title(f"{method_name}")
        axs[idx_m + 2].text(
            0.25,
            0.9,
            "r={:.3f}".format(
                np.corrcoef(xs[block_type], ys[block_type][method_name])[0, 1]
            ),
            ha="center",
            va="center",
            transform=axs[idx_m + 2].transAxes,
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

    with open(os.path.join(RESULTS_PATH, "scores_wo_pairs.pt"), "rb") as handle:
        scores_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "idxs_wo_pairs.pt"), "rb") as handle:
        idxs_wo = dill.load(handle)

    # median plot
    f, axs = plt.subplots(figsize=(16, 8), nrows=1, ncols=2, sharey=True, tight_layout=True)
    fontP = FontProperties()
    fontP.set_size("medium")
    cm = plt.get_cmap("jet")
    num_colors = ((len(AVAIL_METHODS)) * len(BLOCK_TYPES)) + 3
    axs[0].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])
    axs[1].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])

    axs[0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(scores["rf"]["rf"][idx_th]) * 100) 
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(scores_wo["rf"]["rf"][idx_th]) * 100) 
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(scores["dnn"]["dnn"][idx_th]) * 100) 
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (DNN)",
        marker="o",
    )

    axs[1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(scores_wo["dnn"]["dnn"][idx_th]) * 100) 
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (DNN)",
        marker="o",
    )

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        for method in avail_methods:
            method_name = method if isinstance(method, str) else method.__name__
            medians = [
                np.median(np.array(scores[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
            medians_wo = [
                np.median(np.array(scores_wo[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
            axs[0].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
            axs[1].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_wo,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
        axs[0].grid(True)
        axs[1].grid(True)
    axs[0].set_title(r"Including benchmark pairs in training", fontsize=14)
    axs[1].set_title(r"Excluding benchmark pairs from training", fontsize=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    axs[0].set_ylabel(r"Color accuracy (\%)", fontsize=14)
    xlabel = f.text(0.45, 0.04, r"Minimum shared MCS atoms among pairs (\%)", ha='center', fontsize=14)

    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop=fontP,
        fancybox=True,
        shadow=True,
    )
    plt.subplots_adjust(right=0.75, top=1.46, bottom=0.95)
    plt.savefig(
        os.path.join(FIG_PATH, f"color.pdf"), dpi=300, bbox_extra_artists=(xlabel, legend,), bbox_inches="tight"
    )
    plt.close()

    ## Common variables for other plots

    colors = {}
    colors["rf"] = np.array(
        sorted(glob(os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf_wo_pairs.pt")))
    )[
        idxs["rf"]["rf"][0]
    ]  # 0 is at MCS threshold .5

    colors["dnn"] = np.array(
        sorted(glob(os.path.join(DATA_PATH, "validation_sets", "*", "colors_dnn_wo_pairs.pt")))
    )[
        idxs["dnn"]["dnn"][0]
    ]  # 0 is at MCS threshold .5

    for bt in BLOCK_TYPES:
        colors[bt] = np.array(
            sorted(
                glob(
                    os.path.join(DATA_PATH, "validation_sets", "*", f"colors_{bt}_wo_pairs.pt",)
                )
            )
        )[idxs[bt]["CAM"][0]]

    # similarities
    similarities = collections.defaultdict(list)
    exists = collections.defaultdict(list)

    y = {}

    for idx, color_f in enumerate(tqdm(colors["rf"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
        if os.path.exists(sim_file):
            similarities["rf"].append(np.load(sim_file).mean())
            exists["rf"].append(idx)

    y["rf"] = np.array(scores["rf"]["rf"][0])[exists["rf"]]

    for idx, color_f in enumerate(tqdm(colors["dnn"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity.npy")
        if os.path.exists(sim_file):
            similarities["dnn"].append(np.load(sim_file).mean())
            exists["dnn"].append(idx)

    y["dnn"] = np.array(scores["dnn"]["dnn"][0])[exists["dnn"]]

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
            savename="similarity_wo_pairs",
        )

    # training set size
    sizes = collections.defaultdict(list)
    exists = collections.defaultdict(list)
    y = {}

    for idx, color_f in enumerate(colors["rf"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(sim_file):
            sizes["rf"].append(len(pd.read_csv(train_file)))
            exists["rf"].append(idx)

    y["rf"] = np.array(scores["rf"]["rf"][0])[exists["rf"]]

    for idx, color_f in enumerate(colors["dnn"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(sim_file):
            sizes["dnn"].append(len(pd.read_csv(train_file)))
            exists["dnn"].append(idx)

    y["dnn"] = np.array(scores["dnn"]["dnn"][0])[exists["dnn"]]

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
            savename="sizes_wo_pairs",
        )

    # performance
    all_metrics = {}
    all_metrics["rmse_train"] = collections.defaultdict(list)
    all_metrics["rmse_test"] = collections.defaultdict(list)
    all_metrics["pcc_train"] = collections.defaultdict(list)
    all_metrics["pcc_test"] = collections.defaultdict(list)

    exists = collections.defaultdict(list)

    y = {}

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

    y["rf"] = np.array(scores["rf"]["rf"][0])[exists["rf"]]

    for idx, color_f in enumerate(tqdm(colors["dnn"])):
        id_ = os.path.basename(os.path.dirname(color_f))
        metrics_path = os.path.join(LOG_PATH, f"{id_}_metrics_dnn.pt")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as handle:
                metrics = dill.load(handle)
                all_metrics["rmse_train"]["dnn"].append(metrics["rmse_train"])
                all_metrics["rmse_test"]["dnn"].append(metrics["rmse_test"])
                all_metrics["pcc_train"]["dnn"].append(metrics["pcc_train"])
                all_metrics["pcc_test"]["dnn"].append(metrics["pcc_test"])
            exists["dnn"].append(idx)

    y["dnn"] = np.array(scores["dnn"]["dnn"][0])[exists["dnn"]]

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
            savename="train_rmse_wo_pairs",
        )
        comparison_plot(
            all_metrics["rmse_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test RMSE",
            savename="test_rmse_wo_pairs",
        )
        comparison_plot(
            all_metrics["pcc_train"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train PCC",
            savename="train_pcc_wo_pairs",
        )
        comparison_plot(
            all_metrics["pcc_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test PCC",
            savename="test_pcc_wo_pairs",
        )

    # overlap
    with open(os.path.join(RESULTS_PATH, "train_pairs_overlap.pt"), "rb") as handle:
        overlap = dill.load(handle)

    plt.hist(overlap.values(), bins=30)
    plt.xlabel("Fraction of benchmark compounds present in the training sets")
    plt.ylabel("Count")
    plt.savefig(os.path.join(FIG_PATH, "overlap.png"))
    plt.close()
