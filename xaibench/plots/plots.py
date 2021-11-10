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

FONTSIZE = 14
THRESHOLD_IDX = 0


def comparison_plot(xs, ys, block_type, avail_methods, common_x_label, savename):
    ncols = len(avail_methods) + 2  # +2 for sheridan rf, dnn
    f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4), sharey=True)
    axs[0].scatter(xs["rf"], ys["rf"], s=1.5)
    axs[0].set_title(r"Sheridan (RF)", fontsize=FONTSIZE)
    axs[0].set_ylabel(r"Color accuracy (\%)", fontsize=FONTSIZE)
    axs[0].text(
        0.3,
        0.9,
        "PCC={:.3f}".format(np.corrcoef(xs["rf"], ys["rf"])[0, 1]),
        va="center",
        ha="center",
        transform=axs[0].transAxes,
        bbox={'facecolor': 'skyblue', 'alpha': 0.9, 'pad': 5}
    )
    axs[0].tick_params(labelsize=FONTSIZE)

    axs[1].scatter(xs["dnn"], ys["dnn"], s=1.5)
    axs[1].set_title(r"Sheridan (DNN)", fontsize=FONTSIZE)
    axs[1].text(
        0.3,
        0.9,
        "PCC={:.3f}".format(np.corrcoef(xs["dnn"], ys["dnn"])[0, 1]),
        va="center",
        ha="center",
        transform=axs[1].transAxes,
        bbox={'facecolor': 'skyblue', 'alpha': 0.9, 'pad': 5}
    )
    axs[1].tick_params(labelsize=FONTSIZE)


    for idx_m, method in enumerate(avail_methods):
        method_name = method if isinstance(method, str) else method.__name__

        axs[idx_m + 2].scatter(xs[block_type], ys[block_type][method_name], s=1.5)
        axs[idx_m + 2].set_title(f"{method_name}", fontsize=FONTSIZE)
        axs[idx_m + 2].text(
            0.3,
            0.9,
            "PCC={:.3f}".format(
                np.corrcoef(xs[block_type], ys[block_type][method_name])[0, 1]
            ),
            ha="center",
            va="center",
            transform=axs[idx_m + 2].transAxes,
            bbox={'facecolor': 'skyblue', 'alpha': 0.9, 'pad': 5}
        )
        axs[idx_m + 2].tick_params(labelsize=FONTSIZE)

    f.text(0.5, 0.0, common_x_label, ha="center", fontsize=FONTSIZE)
    # plt.suptitle(f"Block type: {block_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, f"{savename}_{block_type}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    with open(os.path.join(RESULTS_PATH, "accs.pt"), "rb") as handle:
        accs = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "f1s.pt"), "rb") as handle:
        f1s = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "directions.pt"), "rb") as handle:
        directions = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "idxs.pt"), "rb") as handle:
        idxs = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "accs_wo_pairs.pt"), "rb") as handle:
        accs_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "f1s_wo_pairs.pt"), "rb") as handle:
        f1s_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "directions_wo_pairs.pt"), "rb") as handle:
        directions_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "idxs_wo_pairs.pt"), "rb") as handle:
        idxs_wo = dill.load(handle)

    # median plot
    f, axs = plt.subplots(
        figsize=(16, 16), nrows=2, ncols=2, sharey="row", sharex=True, tight_layout=True
    )
    fontP = FontProperties()
    fontP.set_size(12)
    cm = plt.get_cmap("jet")
    num_colors = ((len(AVAIL_METHODS)) * len(BLOCK_TYPES)) + 3
    axs[0, 0].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])
    axs[0, 1].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])
    axs[1, 0].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])
    axs[1, 1].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])

    ## accs rf & dnn

    axs[0, 0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(accs["rf"]["rf"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[0, 1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(accs_wo["rf"]["rf"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[0, 0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(accs["dnn"]["dnn"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (DNN)",
        marker="o",
    )

    axs[0, 1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(accs_wo["dnn"]["dnn"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (DNN)",
        marker="o",
    )

    ## f1s rf & dnn

    axs[1, 0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(f1s["rf"]["rf"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[1, 1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(f1s_wo["rf"]["rf"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[1, 0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(f1s["dnn"]["dnn"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (DNN)",
        marker="o",
    )

    axs[1, 1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(f1s_wo["dnn"]["dnn"][idx_th]) * 100)
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
            medians_acc = [
                np.median(np.array(accs[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
            medians_acc_wo = [
                np.median(np.array(accs_wo[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]

            medians_f1 = [
                np.median(np.array(f1s[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
            medians_f1_wo = [
                np.median(np.array(f1s_wo[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]

            axs[0, 0].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_acc,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
            axs[0, 1].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_acc_wo,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
            axs[1, 0].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_f1,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
            axs[1, 1].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_f1_wo,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
        axs[0, 0].grid(True)
        axs[0, 1].grid(True)
        axs[1, 0].grid(True)
        axs[1, 1].grid(True)
    axs[0, 0].set_title(r"Including benchmark pairs in training", fontsize=14)
    axs[0, 1].set_title(r"Excluding benchmark pairs from training", fontsize=14)
    axs[0, 0].tick_params(labelsize=14)
    axs[0, 1].tick_params(labelsize=14)
    axs[1, 0].tick_params(labelsize=14)
    axs[1, 1].tick_params(labelsize=14)
    axs[0, 0].set_ylabel(r"Color accuracy (\%)", fontsize=14)
    axs[1, 0].set_ylabel(r"Color F1-score (\%)", fontsize=14)
    xlabel = f.text(
        0.45,
        0.04,
        r"Minimum shared MCS atoms among pairs (\%)",
        ha="center",
        fontsize=14,
    )

    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop=fontP,
        fancybox=True,
        shadow=True,
    )
    plt.subplots_adjust(right=0.75, top=1.57, bottom=0.95)
    plt.savefig(
        os.path.join(FIG_PATH, f"color.svg"),
        dpi=300,
        bbox_extra_artists=(xlabel, legend,),
        bbox_inches="tight",
    )
    plt.close()

    ## direction plot

    f, axs = plt.subplots(
        figsize=(16, 16), nrows=2, ncols=2, sharey="row", sharex=True, tight_layout=True
    )
    fontP = FontProperties()
    fontP.set_size(12)
    cm = plt.get_cmap("jet")
    num_colors = ((len(AVAIL_METHODS)) * len(BLOCK_TYPES)) + 3
    axs[0, 0].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])
    axs[0, 1].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])


    axs[0, 0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(directions["rf"]["rf"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[0, 1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(directions_wo["rf"]["rf"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (RF)",
        marker="o",
    )

    axs[0, 0].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(directions["dnn"]["dnn"][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
        ),
        label="Sheridan (DNN)",
        marker="o",
    )

    axs[0, 1].plot(
        MIN_PER_COMMON_ATOMS * 100,
        np.array(
            [
                np.median(np.array(directions_wo["dnn"]["dnn"][idx_th]) * 100)
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
            medians_directions = [
                np.median(np.array(directions[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]
            medians_directions_wo = [
                np.median(np.array(directions_wo[bt][method_name][idx_th]) * 100)
                for idx_th in range(N_THRESHOLDS)
            ]

            axs[0, 0].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_directions,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
            axs[0, 1].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_directions_wo,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
        axs[0, 0].grid(True)
        axs[0, 1].grid(True)

    axs[0, 0].set_title(r"Including benchmark pairs in training", fontsize=14)
    axs[0, 1].set_title(r"Excluding benchmark pairs from training", fontsize=14)
    axs[0, 0].tick_params(labelsize=14)
    axs[0, 1].tick_params(labelsize=14)
    axs[0, 0].set_ylabel(r"Aggregated color direction accuracy (\%)", fontsize=14)

    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop=fontP,
        fancybox=True,
        shadow=True,
    )
    plt.subplots_adjust(right=0.75, top=1.57, bottom=0.95)
    plt.savefig(
        os.path.join(FIG_PATH, f"direction.svg"),
        dpi=300,
        bbox_extra_artists=(xlabel, legend,),
        bbox_inches="tight",
    )
    plt.close()

    ## Common variables for other plots

    colors = {}

    colors["rf"] = np.array(
        sorted(
            glob(
                os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf_wo_pairs.pt")
            )
        )
    )[
        idxs_wo["rf"]["rf"][THRESHOLD_IDX]
    ]  # 0 is at MCS threshold .5

    colors["dnn"] = np.array(
        sorted(
            glob(
                os.path.join(
                    DATA_PATH, "validation_sets", "*", "colors_dnn_wo_pairs.pt"
                )
            )
        )
    )[
        idxs_wo["dnn"]["dnn"][THRESHOLD_IDX]
    ]  # 0 is at MCS threshold .5

    for bt in BLOCK_TYPES:
        colors[bt] = np.array(
            sorted(
                glob(
                    os.path.join(
                        DATA_PATH, "validation_sets", "*", f"colors_{bt}_wo_pairs.pt",
                    )
                )
            )
        )[
            idxs_wo[bt]["CAM"][THRESHOLD_IDX]
        ]  # CAM or other GNN-based method is the same. 0 is at MCS threshold .5

    # similarities
    similarities = collections.defaultdict(list)
    exists = collections.defaultdict(list)

    y = {}

    for idx, color_f in enumerate(tqdm(colors["rf"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity_wo_pairs.npy")
        if os.path.exists(sim_file):
            similarities["rf"].append(np.load(sim_file).mean())
            exists["rf"].append(idx)

    y["rf"] = np.array(accs_wo["rf"]["rf"][0])[exists["rf"]] * 100

    for idx, color_f in enumerate(tqdm(colors["dnn"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity_wo_pairs.npy")
        if os.path.exists(sim_file):
            similarities["dnn"].append(np.load(sim_file).mean())
            exists["dnn"].append(idx)

    y["dnn"] = np.array(accs_wo["dnn"]["dnn"][0])[exists["dnn"]] * 100

    for bt in BLOCK_TYPES:
        print(bt)
        ncols = len(AVAIL_METHODS) + 2 if bt == "gat" else len(AVAIL_METHODS) + 1

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))

        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors[bt])):
            sim_file = os.path.join(os.path.dirname(color_f), "similarity_wo_pairs.npy")
            if os.path.exists(sim_file):
                similarities[bt].append(np.load(sim_file).mean())
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(accs_wo[bt][method_name][0])[exists[bt]] * 100


    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            similarities,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Average Tanimoto similarity between training and benchmark molecules",
            savename="similarity_wo_pairs",
        )

    # training set size
    sizes = collections.defaultdict(list)
    exists = collections.defaultdict(list)
    y = {}

    for idx, color_f in enumerate(colors["rf"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(train_file):
            sizes["rf"].append(len(pd.read_csv(train_file)))
            exists["rf"].append(idx)

    y["rf"] = np.array(accs_wo["rf"]["rf"][0])[exists["rf"]]

    for idx, color_f in enumerate(colors["dnn"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(train_file):
            sizes["dnn"].append(len(pd.read_csv(train_file)))
            exists["dnn"].append(idx)

    y["dnn"] = np.array(accs_wo["dnn"]["dnn"][0])[exists["dnn"]]

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
            y[bt][method_name] = np.array(accs_wo[bt][method_name][0])[exists[bt]]

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

    # assay ids
    nassays = collections.defaultdict(list)
    exists = collections.defaultdict(list)
    y = {}

    for idx, color_f in enumerate(colors["rf"]):
        assay_f = os.path.join(os.path.dirname(color_f), "assay_ids.npy")
        if os.path.exists(assay_f):
            nassays["rf"].append(len(np.load(assay_f)))
            exists["rf"].append(idx)
    
    y["rf"] = np.array(accs_wo["rf"]["rf"][0])[exists["rf"]]


    for idx, color_f in enumerate(colors["dnn"]):
        assay_f = os.path.join(os.path.dirname(color_f), "assay_ids.npy")
        if os.path.exists(assay_f):
            nassays["dnn"].append(len(np.load(assay_f)))
            exists["dnn"].append(idx)
    
    y["dnn"] = np.array(accs_wo["dnn"]["dnn"][0])[exists["dnn"]]


    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors[bt])):
            assay_f = os.path.join(os.path.dirname(color_f), "assay_ids.npy")
            if os.path.exists(assay_f):
                nassays[bt].append(len(np.load(assay_f)))
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(tqdm(avail_methods)):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(accs_wo[bt][method_name][0])[exists[bt]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            nassays,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Number of different training assays",
            savename="nassays_wo_pairs",
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

    y["rf"] = np.array(accs_wo["rf"]["rf"][0])[exists["rf"]]

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

    y["dnn"] = np.array(accs_wo["dnn"]["dnn"][0])[exists["dnn"]]

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
            y[bt][method_name] = np.array(accs_wo[bt][method_name][0])

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
