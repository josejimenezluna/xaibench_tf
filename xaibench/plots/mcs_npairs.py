import os
from glob import glob

import dill
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
from tqdm import tqdm
from xaibench.determine_col import MIN_PER_COMMON_ATOMS
from xaibench.score import N_THRESHOLDS
from xaibench.utils import BENCHMARK_PATH, FIG_PATH

matplotlib.use("Agg")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

if __name__ == "__main__":
    npairs_threshold = np.zeros(N_THRESHOLDS)

    color_fs = glob(os.path.join(BENCHMARK_PATH, "*", "colors.pt"))

    for color_f in tqdm(color_fs):
        with open(color_f, "rb") as handle:
            colors = dill.load(handle)

        n_pairs = np.zeros(N_THRESHOLDS)
        for pair in colors:
            n_pairs += np.array([1 if p is not None else 0 for p in pair])
        npairs_threshold += n_pairs

    f, ax = plt.subplots(figsize=(4, 4), nrows=1, ncols=1, tight_layout=True)
    fontP = FontProperties()
    fontP.set_size(12)

    ax.plot(
        MIN_PER_COMMON_ATOMS * 100,
        npairs_threshold,
        marker="o",
        color="black"
    )
    ylabels = [r'{:d}'.format(int(x)) + r'K' for x in ax.get_yticks() / 1000]
    ax.set_yticklabels(ylabels)

    ax.set_xlabel(r"Minimum shared MCS atoms among pairs (\%)")
    ax.set_ylabel(r"Number of benchmark pairs")
    ax.grid()

    plt.savefig(
        os.path.join(FIG_PATH, "npairs_mcs.pdf"),
    )
    plt.close()