import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.utils import DATA_PATH, FIG_PATH, LOG_PATH

VERSION_MOLGRAD = 2
N_THRESHOLDS = len(MIN_PER_COMMON_ATOMS)


def color_agreement(color_true, color_pred):
    idx_noncommon = [idx for idx, val in color_true.items() if val != 0.0]
    if len(idx_noncommon) == 0:
        return -1.0
    color_true_noncommon = np.array([color_true[idx] for idx in idx_noncommon])
    color_pred_noncommon = np.sign([color_pred[idx] for idx in idx_noncommon])
    agreement = color_true_noncommon == color_pred_noncommon
    return np.mean(agreement)


def method_comparison(colors_method, idx_threshold):
    avg_scores = []
    idx_valid = []

    for idx, color_method_f in enumerate(tqdm(colors_method)):
        dirname = os.path.dirname(color_method_f)
        with open(os.path.join(dirname, "colors.pt"), "rb") as handle:
            colors = pickle.load(handle)

        colors = [col[idx_threshold] for col in colors]

        with open(color_method_f, "rb",) as handle:
            col_method = pickle.load(handle)

        if sum(1 for _ in filter(None.__ne__, colors)) > 0:
            scores = []
            for color_pair_true, color_pair_pred in zip(colors, col_method):
                if color_pair_true is not None:
                    ag_i = color_agreement(color_pair_true[0], color_pair_pred[0])
                    scores.append(ag_i)
                    ag_j = color_agreement(color_pair_true[1], color_pair_pred[1])
                    scores.append(ag_j)

            scores = np.array(scores)
            scores = scores[scores >= 0.0]

            if scores.size > 0:
                avg_scores.append(scores.mean())
                idx_valid.append(idx)
    return np.array(avg_scores), set(idx_valid)


if __name__ == "__main__":
    colors_molgrad_all = glob(
        os.path.join(
            DATA_PATH,
            "validation_sets",
            "*",
            f"colors_molgrad_noglobal_{VERSION_MOLGRAD}.pt",
        )
    )

    colors_rf_all = glob(
        os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf.pt")
    )

    medians_molgrad = []
    stds_molgrad = []
    medians_rf = []
    stds_rf = []

    for idx_th in range(N_THRESHOLDS):
        print(f"Computing threshold {idx_th + 1}/{N_THRESHOLDS}...")
        scores_molgrad, idx_valid_molgrad = method_comparison(
            colors_molgrad_all, idx_th
        )
        scores_rf, idx_valid_rf = method_comparison(colors_rf_all, idx_th)

        f, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].hist(scores_molgrad, bins=50)
        m_molgrad = np.median(scores_molgrad)
        medians_molgrad.append(m_molgrad)
        stds_molgrad.append(np.std(scores_molgrad))
        axs[0].axvline(m_molgrad, linestyle="--", color="black")
        axs[1].hist(scores_rf, bins=50)
        m_rf = np.median(scores_rf)
        medians_rf.append(m_rf)
        stds_rf.append(np.std(scores_rf))
        axs[1].axvline(m_rf, linestyle="--", color="black")
        axs[0].set_xlabel("IG")
        axs[1].set_xlabel("Sheridan")

        plt.suptitle("Average agreement between attributions and coloring")
        plt.savefig(
            os.path.join(FIG_PATH, "color_agreement_noglobal_{}.png".format(idx_th)),
            dpi=300,
        )
        plt.close()

    f, ax = plt.subplots()
    eb1 = ax.errorbar(
        MIN_PER_COMMON_ATOMS, medians_molgrad, yerr=stds_molgrad, label="IG"
    )
    eb2 = ax.errorbar(
        MIN_PER_COMMON_ATOMS, medians_rf, yerr=stds_rf, label="Sheridan", alpha=0.5
    )
    eb1[-1][0].set_linestyle("--")
    eb2[-1][0].set_linestyle("--")
    ax.axhline(0.5, label="Random", linestyle="--", c="black")
    ax.grid()
    ax.set_xlabel("Minimum percentage of common atoms in MCS")
    ax.set_ylabel("Median agreement")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(FIG_PATH, "medianagreement_noglobal.png"), dpi=300)
    plt.close()

    colors_molgrad_all = [
        c for idx, c in enumerate(colors_molgrad_all) if idx in idx_valid_molgrad
    ]
    colors_rf_all = [c for idx, c in enumerate(colors_rf_all) if idx in idx_valid_rf]

    # TODO: these plots need to be redone

    # similarities
    exist_idx_sim_molgrad = []
    similarities_molgrad = []

    for idx, c in enumerate(colors_molgrad_all):
        sim_file = os.path.join(os.path.dirname(c), "similarity.npy")
        if os.path.exists(sim_file):
            similarities_molgrad.append(sim_file)
            exist_idx_sim_molgrad.append(idx)

    exist_idx_sim_rf = []
    similarities_rf = []

    for idx, c in enumerate(colors_rf_all):
        sim_file = os.path.join(os.path.dirname(c), "similarity.npy")
        if os.path.exists(sim_file):
            similarities_rf.append(sim_file)
            exist_idx_sim_rf.append(idx)

    avg_sim_molgrad = []
    max_sim_molgrad = []

    for sim_fs in similarities_molgrad:
        sim = np.load(sim_fs)
        avg_sim_molgrad.append(sim.mean())
        max_sim_molgrad.append(sim.max())

    avg_sim_rf = []
    max_sim_rf = []

    for sim_fs in similarities_rf:
        sim = np.load(sim_fs)
        avg_sim_rf.append(sim.mean())
        max_sim_rf.append(sim.max())

    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].scatter(avg_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad], s=1.5)
    axs[1].scatter(avg_sim_rf, scores_rf[exist_idx_sim_rf], s=1.5)
    axs[0].set_ylabel("Agreement between attributions and coloring")
    axs[0].set_xlabel("Avg. Tanimoto similarities between train and test sets")
    axs[0].set_title("IG")
    axs[0].text(
        0.3,
        0.9,
        "r={:.3f}".format(
            np.corrcoef(avg_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad])[0, 1]
        ),
        style="italic",
        bbox={"facecolor": "red", "alpha": 0.5},
    )
    axs[1].set_ylabel("Agreement between attributions and coloring")
    axs[1].set_xlabel("Avg. Tanimoto similarities between train and test sets")
    axs[1].set_title("Sheridan")
    axs[1].text(
        0.3,
        0.9,
        "r={:.3f}".format(np.corrcoef(avg_sim_rf, scores_rf[exist_idx_sim_rf])[0, 1]),
        style="italic",
        bbox={"facecolor": "red", "alpha": 0.5},
    )
    plt.savefig(os.path.join(FIG_PATH, "avgsimilarityvsagreement.png"))
    plt.close()

    # max
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].scatter(max_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad], s=1.5)
    axs[1].scatter(max_sim_rf, scores_rf[exist_idx_sim_rf], s=1.5)
    axs[0].set_ylabel("Agreement between attributions and coloring")
    axs[0].set_xlabel("Max. Tanimoto similarities between train and test sets")
    axs[0].set_title("IG")
    axs[0].text(
        0.3,
        0.9,
        "r={:.3f}".format(
            np.corrcoef(max_sim_molgrad, scores_molgrad[exist_idx_sim_molgrad])[0, 1]
        ),
        style="italic",
        bbox={"facecolor": "red", "alpha": 0.5},
    )
    axs[1].set_ylabel("Agreement between attributions and coloring")
    axs[1].set_xlabel("Max. Tanimoto similarities between train and test sets")
    axs[1].set_title("Sheridan")
    axs[1].text(
        0.3,
        0.9,
        "r={:.3f}".format(np.corrcoef(max_sim_rf, scores_rf[exist_idx_sim_rf])[0, 1]),
        style="italic",
        bbox={"facecolor": "red", "alpha": 0.5},
    )
    plt.savefig(os.path.join(FIG_PATH, "maxsimilarityvsagreement.png"))
    plt.close()

    # performance
    exist_idx_log_molgrad = []
    rs_molgrad = []

    for idx, c in enumerate(colors_molgrad_all):
        log_file = os.path.join(
            LOG_PATH, f"{os.path.basename(os.path.dirname(c))}_metrics.pt"
        )
        if os.path.exists(log_file):
            with open(log_file, "rb") as handle:
                metrics = pickle.load(handle)
                r = metrics[1][0]
            rs_molgrad.append(r)
            exist_idx_log_molgrad.append(idx)

    exist_idx_log_rf = []
    rs_rf = []

    for idx, c in enumerate(colors_molgrad_all):
        log_file = os.path.join(
            LOG_PATH, f"{os.path.basename(os.path.dirname(c))}_metrics_rf.pt"
        )
        if os.path.exists(log_file):
            with open(log_file, "rb") as handle:
                metrics = pickle.load(handle)
                r = metrics[1]
            rs_rf.append(r)
            exist_idx_log_rf.append(idx)

    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].scatter(rs_molgrad, scores_molgrad[exist_idx_log_molgrad], s=1.5)
    axs[1].scatter(rs_rf, scores_rf[exist_idx_log_rf], s=1.5)
    axs[0].set_ylabel("Agreement between attributions and coloring")
    axs[0].set_xlabel("Correlation on held-out test set")
    axs[0].set_title("IG")
    axs[1].set_ylabel("Agreement between attributions and coloring")
    axs[1].set_xlabel("Correlation on held-out test set")
    axs[1].set_title("Sheridan")
    plt.savefig(os.path.join(FIG_PATH, "performancevsagreement.png"))
    plt.close()
