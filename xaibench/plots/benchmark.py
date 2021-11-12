import os

import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.score import N_THRESHOLDS
from xaibench.utils import BLOCK_TYPES, FIG_PATH, RESULTS_PATH

matplotlib.use("Agg")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

FONTSIZE = 14


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
        os.path.join(FIG_PATH, "color.svg"),
        dpi=300,
        bbox_extra_artists=(xlabel, legend,),
        bbox_inches="tight",
    )
    plt.close()

    ## direction plot

    f, axs = plt.subplots(
        figsize=(16, 9), nrows=1, ncols=2, sharey="row", sharex=True, tight_layout=True
    )
    fontP = FontProperties()
    fontP.set_size(12)
    cm = plt.get_cmap("jet")
    num_colors = ((len(AVAIL_METHODS)) * len(BLOCK_TYPES)) + 3
    axs[0].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])
    axs[1].set_prop_cycle("color", [cm(i / num_colors) for i in range(num_colors)])


    axs[0].plot(
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

    axs[1].plot(
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

    axs[0].plot(
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

    axs[1].plot(
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

            axs[0].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_directions,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
            axs[1].plot(
                MIN_PER_COMMON_ATOMS * 100,
                medians_directions_wo,
                label=f"{bt.upper()} ({method_name})",
                marker="o",
            )
        axs[0].grid(True)
        axs[1].grid(True)

    axs[0].set_title(r"Including benchmark pairs in training", fontsize=14)
    axs[1].set_title(r"Excluding benchmark pairs from training", fontsize=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    axs[0].set_ylabel(r"Aggregated color direction accuracy (\%)", fontsize=14)
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
        os.path.join(FIG_PATH, f"direction.svg"),
        dpi=300,
        bbox_extra_artists=(xlabel, legend,),
        bbox_inches="tight",
    )
    plt.close()

