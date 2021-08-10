import os
import numpy as np
import dill
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from xaibench.utils import FIG_PATH, RESULTS_PATH

matplotlib.use("Agg")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

if __name__ == "__main__":
    with open(os.path.join(RESULTS_PATH, "method_agreement.pt"), "rb") as handle:
        agreement = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "method_keys.pt"), "rb") as handle:
        method_names = dill.load(handle)

    agreement = np.dstack([ag for ag in agreement if ag is not None]).mean(axis=2)

    mask = np.zeros_like(agreement)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(18, 16))
        ax = sns.heatmap(
            agreement,
            mask=mask,
            cbar=True,
            square=True,
            linewidths=0.5,
            xticklabels=method_names,
            yticklabels=method_names,
            cbar_kws={"shrink": .5}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=15, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=15)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18, left=True)
        cbar.ax.set_ylabel(r"Avg. Spearman's $\rho$", rotation=270, size=18)
        cbar.ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "agreement.pdf"), dpi=300)
    plt.close()
