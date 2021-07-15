import os
import dill
from glob import glob
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xaibench.utils import BLOCK_TYPES, DATA_PATH, FIG_PATH, LOG_PATH
from xaibench.plots.desc import BINSIZE

matplotlib.use("Agg")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

if __name__ == "__main__":
    rmses = {}
    rs = {}

    for bt in BLOCK_TYPES:
        logs = glob(os.path.join(LOG_PATH, f"{bt}_*_wo_pairs.pt"))
        print(len(logs))
        for log in tqdm(logs):
            with open(log, "rb") as handle:
                perf = dill.load(handle)
            rmses.setdefault(bt, []).append(perf["rmse_test"][-1])
            rs.setdefault(bt, []).append(perf["pcc_test"][-1])

    logs_rf = glob(os.path.join(LOG_PATH, f"*_rf_wo_pairs.pt"))
    print(len(logs_rf))

    rmses["rf"] = []
    rs["rf"] = []

    for log in tqdm(logs_rf):
        with open(log, "rb") as handle:
            perf = dill.load(handle)

            rmses["rf"].append(perf["rmse_test"])
            rs["rf"].append(perf["pcc_test"])

    logs_dnn = glob(os.path.join(LOG_PATH, f"*_dnn_wo_pairs.pt"))
    print(len(logs_dnn))

    rmses["dnn"] = []
    rs["dnn"] = []

    for log in tqdm(logs_dnn):
        with open(log, "rb") as handle:
            perf = dill.load(handle)

            rmses["dnn"].append(perf["rmse_test"])
            rs["dnn"].append(perf["pcc_test"])

    f, axs = plt.subplots(
        nrows=2, ncols=6, sharex="row", sharey="row", figsize=(16, 8), tight_layout=True
    )
    for idx, bt in enumerate(["rf", "dnn"] + BLOCK_TYPES):
        axs[0, idx].hist(rmses[bt], bins=BINSIZE)
        med_rmse = np.median(rmses[bt])
        med_pcc = np.median(rs[bt])
        axs[0, idx].axvline(med_rmse, linestyle="--", c="black")
        axs[0, idx].set_title(bt.upper())

        axs[0, idx].text(
            0.7,
            0.9,
            "RMSE={:.3f}".format(med_rmse),
            va="center",
            ha="center",
            transform=axs[0, idx].transAxes,
            fontsize=14
        )
        axs[0, idx].tick_params(labelsize=14)


        axs[1, idx].hist(rs[bt], bins=BINSIZE, facecolor="orange")
        axs[1, idx].axvline(med_pcc, linestyle="--", c="black")

        axs[1, idx].text(
            0.25,
            0.9,
            "PCC={:.3f}".format(med_pcc),
            va="center",
            ha="center",
            transform=axs[1, idx].transAxes,
            fontsize=14,
        )
        axs[1, idx].tick_params(labelsize=14)


    # axs[0, 0].set_ylabel("RMSE", fontsize=14)
    # axs[1, 0].set_ylabel("PCC", fontsize=14)

    plt.savefig(os.path.join(FIG_PATH, "perf.pdf"), dpi=300)
    plt.close()
