import os
from glob import glob

import dill
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from xaibench.color import AVAIL_METHODS
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.utils import DATA_PATH, FIG_PATH, LOG_PATH

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


def method_comparison(all_colors_method, idx_threshold, method):
    avg_scores = []
    idx_valid = []

    for idx, color_method_f in enumerate(tqdm(all_colors_method)):
        dirname = os.path.dirname(color_method_f)

        with open(os.path.join(dirname, "colors.pt"), "rb") as handle:
            colors = dill.load(handle)

        colors = [col[idx_threshold] for col in colors]

        with open(color_method_f, "rb") as handle:
            colors_method = dill.load(handle)

        if method in AVAIL_METHODS:
            colors_method = [
                (cm[0].nodes.numpy(), cm[1].nodes.numpy())
                for cm in colors_method[method.__name__]
            ]

        if sum(1 for _ in filter(None.__ne__, colors)) > 0:
            scores = []
            for color_pair_true, color_pair_pred in zip(colors, colors_method):
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
    colors_method = glob(
        os.path.join(DATA_PATH, "validation_sets", "*", f"colors_gcn.pt",)
    )

    colors_rf_all = glob(
        os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf.pt")
    )

    for idx_th in range(N_THRESHOLDS):
        print(f"Computing threshold {idx_th + 1}/{N_THRESHOLDS}...")

        scores_rf, idx_valid_rf = method_comparison(colors_rf_all, idx_th, method="rf")

        f, axs = plt.subplots(nrows=1, ncols=4)

        for idx_method, method in enumerate(AVAIL_METHODS):
            scores_method, idx_valid_method = method_comparison(
                colors_method, idx_th, method=method
            )
            axs[idx_method].hist(scores_method, bins=50)
            axs[idx_method].axvline(
                np.median(scores_method), linestyle="--", color="black"
            )
            axs[idx_method].set_xlabel(method.__name__)

        axs[3].hist(scores_rf, bins=50)
        axs[3].axvline(np.median(scores_rf), linestyle="--", color="black")
        axs[3].set_xlabel("Sheridan")

        plt.suptitle("Average agreement between attributions and coloring")
        plt.show()
        plt.savefig(
            os.path.join(FIG_PATH, "color_agreement_all_{}.png".format(idx_th)),
            dpi=300,
        )
        plt.close()
