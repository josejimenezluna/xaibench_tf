import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xaibench.utils import DATA_PATH, FIG_PATH, RESULTS_PATH

LIMIT_PSIZE = 5000
BINSIZE = 50

matplotlib.use("Agg")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)


if __name__ == "__main__":
    pairs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "pairs.csv"))
    psizes = np.array([len(pd.read_csv(p)) for p in pairs])

    overlaps = np.load(os.path.join(RESULTS_PATH, "overlap.npy"))

    tsizes = []
    tsizes_wopairs = []

    exists_idx = []
    exists_wo_idx = []

    for idx, pair in enumerate(pairs):
        t_path = os.path.join(os.path.dirname(pair), "training.csv")
        if os.path.exists(t_path):
            train_df = pd.read_csv(t_path)
            tsizes.append(len(train_df))
            exists_idx.append(idx)

        two_path = os.path.join(os.path.dirname(pair), "training_wo_pairs.csv")
        if os.path.exists(two_path):
            train_df = pd.read_csv(two_path)
            tsizes_wopairs.append(len(train_df))
            exists_wo_idx.append(idx)

    tsizes = np.array(tsizes)
    twosizes = np.array(tsizes_wopairs)

    f, axs = plt.subplots(nrows=3, ncols=1, tight_layout=True, figsize=(4, 10))
    axs[0].hist(psizes[psizes < LIMIT_PSIZE], bins=BINSIZE)
    xlabels = [r'{:d}'.format(int(x)) + r'K' for x in axs[0].get_xticks() / 1000]
    axs[0].set_xlabel(r"Benchmark pairs per target")
    axs[0].set_xticklabels(xlabels)

    axs[1].hist(tsizes, bins=BINSIZE)
    xlabels = [r'{:d}'.format(int(x)) + r'K' for x in axs[1].get_xticks() / 1000]
    axs[1].hist(twosizes, bins=BINSIZE, alpha=0.5, label=r"Excl. bench. compounds")
    axs[1].set_xlabel(r"Training samples per target")
    axs[1].set_xticklabels(xlabels)
    axs[1].legend()

    axs[2].hist(np.array(overlaps) * 100, bins=BINSIZE)
    axs[2].set_xlabel(r"Benchmark compounds present in the training sets (\%)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "pairstraining.pdf"))
    plt.close()
