import os
import collections
from glob import glob

import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from xaibench.color import AVAIL_METHODS
from xaibench.utils import BLOCK_TYPES, DATA_PATH, FIG_PATH, LOG_PATH, RESULTS_PATH

matplotlib.use("Agg")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

FONTSIZE = 14
THRESHOLD_IDX = 0

def comparison_plot(xs, ys, block_type, avail_methods, common_x_label, ylabel, savename):
    ncols = len(avail_methods) + 2  # +2 for sheridan rf, dnn
    f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4), sharey=True)
    axs[0].scatter(xs["rf"], ys["rf"], s=1.5)
    axs[0].set_title(r"Sheridan (RF)", fontsize=FONTSIZE)
    axs[0].set_ylabel(ylabel, fontsize=FONTSIZE)
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
    with open(os.path.join(RESULTS_PATH, "accs_wo_pairs.pt"), "rb") as handle:
        accs_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "f1s_wo_pairs.pt"), "rb") as handle:
        f1s_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "directions_wo_pairs.pt"), "rb") as handle:
        directions_wo = dill.load(handle)

    with open(os.path.join(RESULTS_PATH, "idxs_wo_pairs.pt"), "rb") as handle:
        idxs_wo = dill.load(handle)


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

    colors_non_indexed = {}

    colors_non_indexed["rf"] = np.array(
        sorted(
            glob(
                os.path.join(DATA_PATH, "validation_sets", "*", "colors_rf_wo_pairs.pt")
            )
        )
    )

    colors_non_indexed["dnn"] = np.array(
        sorted(
            glob(
                os.path.join(DATA_PATH, "validation_sets", "*", "colors_dnn_wo_pairs.pt")
            )
        )
    )

    for bt in BLOCK_TYPES:
        colors_non_indexed[bt] = np.array(
            sorted(
                glob(
                    os.path.join(
                        DATA_PATH, "validation_sets", "*", f"colors_{bt}_wo_pairs.pt",
                    )
                )
            )
        )

    ## Influence of variables on color accuracy

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
            ylabel=r"Color accuracy (\%)",
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
            ylabel=r"Color accuracy (\%)",
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
            ylabel=r"Color accuracy (\%)",
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
            ylabel=r"Color accuracy (\%)",
            savename="train_rmse_wo_pairs",
        )
        comparison_plot(
            all_metrics["rmse_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test RMSE",
            ylabel=r"Color accuracy (\%)",
            savename="test_rmse_wo_pairs",
        )
        comparison_plot(
            all_metrics["pcc_train"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train PCC",
            ylabel=r"Color accuracy (\%)",
            savename="train_pcc_wo_pairs",
        )
        comparison_plot(
            all_metrics["pcc_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test PCC",
            ylabel=r"Color accuracy (\%)",
            savename="test_pcc_wo_pairs",
        )


    ## Influence of variables on color direction

    # similarities
    similarities = collections.defaultdict(list)
    exists = collections.defaultdict(list)

    y = {}

    for idx, color_f in enumerate(tqdm(colors_non_indexed["rf"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity_wo_pairs.npy")
        if os.path.exists(sim_file):
            similarities["rf"].append(np.load(sim_file).mean())
            exists["rf"].append(idx)

    y["rf"] = np.array(directions_wo["rf"]["rf"][0])[exists["rf"]] * 100

    for idx, color_f in enumerate(tqdm(colors_non_indexed["dnn"])):
        sim_file = os.path.join(os.path.dirname(color_f), "similarity_wo_pairs.npy")
        if os.path.exists(sim_file):
            similarities["dnn"].append(np.load(sim_file).mean())
            exists["dnn"].append(idx)

    y["dnn"] = np.array(directions_wo["dnn"]["dnn"][0])[exists["dnn"]] * 100

    for bt in BLOCK_TYPES:
        print(bt)
        ncols = len(AVAIL_METHODS) + 2 if bt == "gat" else len(AVAIL_METHODS) + 1

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))

        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors_non_indexed[bt])):
            sim_file = os.path.join(os.path.dirname(color_f), "similarity_wo_pairs.npy")
            if os.path.exists(sim_file):
                similarities[bt].append(np.load(sim_file).mean())
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(avail_methods):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(directions_wo[bt][method_name][0])[exists[bt]] * 100


    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            similarities,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Average Tanimoto similarity between training and benchmark molecules",
            ylabel=r"Aggregated direction color accuracy (\%)",
            savename="similarity_direction_wo_pairs",
        )

    # training set size
    sizes = collections.defaultdict(list)
    exists = collections.defaultdict(list)
    y = {}

    for idx, color_f in enumerate(colors_non_indexed["rf"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(train_file):
            sizes["rf"].append(len(pd.read_csv(train_file)))
            exists["rf"].append(idx)

    y["rf"] = np.array(directions_wo["rf"]["rf"][0])[exists["rf"]]

    for idx, color_f in enumerate(colors_non_indexed["dnn"]):
        train_file = os.path.join(os.path.dirname(color_f), "training.csv")
        if os.path.exists(train_file):
            sizes["dnn"].append(len(pd.read_csv(train_file)))
            exists["dnn"].append(idx)

    y["dnn"] = np.array(directions_wo["dnn"]["dnn"][0])[exists["dnn"]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors_non_indexed[bt])):
            train_file = os.path.join(os.path.dirname(color_f), "training.csv")
            if os.path.exists(train_file):
                sizes[bt].append(len(pd.read_csv(train_file)))
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(tqdm(avail_methods)):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(directions_wo[bt][method_name][0])[exists[bt]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            sizes,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Number of training samples",
            ylabel=r"Aggregated direction color accuracy (\%)",
            savename="sizes_direction_wo_pairs",
        )

    # assay_ids

    nassays = collections.defaultdict(list)
    exists = collections.defaultdict(list)
    y = {}

    for idx, color_f in enumerate(colors_non_indexed["rf"]):
        assay_f = os.path.join(os.path.dirname(color_f), "assay_ids.npy")
        if os.path.exists(assay_f):
            nassays["rf"].append(len(np.load(assay_f)))
            exists["rf"].append(idx)
    
    y["rf"] = np.array(directions_wo["rf"]["rf"][0])[exists["rf"]]


    for idx, color_f in enumerate(colors_non_indexed["dnn"]):
        assay_f = os.path.join(os.path.dirname(color_f), "assay_ids.npy")
        if os.path.exists(assay_f):
            nassays["dnn"].append(len(np.load(assay_f)))
            exists["dnn"].append(idx)
    
    y["dnn"] = np.array(directions_wo["dnn"]["dnn"][0])[exists["dnn"]]


    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        for idx, color_f in enumerate(tqdm(colors_non_indexed[bt])):
            assay_f = os.path.join(os.path.dirname(color_f), "assay_ids.npy")
            if os.path.exists(assay_f):
                nassays[bt].append(len(np.load(assay_f)))
                exists[bt].append(idx)

        y[bt] = {}

        for idx_m, method in enumerate(tqdm(avail_methods)):
            method_name = method if isinstance(method, str) else method.__name__
            y[bt][method_name] = np.array(directions_wo[bt][method_name][0])[exists[bt]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]
        comparison_plot(
            nassays,
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Number of different training assays",
            ylabel=r"Aggregated direction color accuracy (\%)",
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

    for idx, color_f in enumerate(tqdm(colors_non_indexed["rf"])):
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

    y["rf"] = np.array(directions_wo["rf"]["rf"][0])[exists["rf"]]

    for idx, color_f in enumerate(tqdm(colors_non_indexed["dnn"])):
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

    y["dnn"] = np.array(directions_wo["dnn"]["dnn"][0])[exists["dnn"]]

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 4))

        for color_f in tqdm(colors_non_indexed[bt]):
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
            y[bt][method_name] = np.array(directions_wo[bt][method_name][0])

    for bt in BLOCK_TYPES:
        avail_methods = AVAIL_METHODS if bt == "gat" else AVAIL_METHODS[:-1]
        avail_methods = avail_methods + ["diff"]

        comparison_plot(
            all_metrics["rmse_train"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train RMSE",
            ylabel=r"Aggregated direction color accuracy (\%)",
            savename="train_rmse_direction_wo_pairs",
        )
        comparison_plot(
            all_metrics["rmse_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test RMSE",
            ylabel=r"Aggregated direction color accuracy (\%)",
            savename="test_rmse_direction_wo_pairs",
        )
        comparison_plot(
            all_metrics["pcc_train"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Train PCC",
            ylabel=r"Aggregated direction color accuracy (\%)",
            savename="train_pcc_direction_wo_pairs",
        )
        comparison_plot(
            all_metrics["pcc_test"],
            y,
            block_type=bt,
            avail_methods=avail_methods,
            common_x_label="Test PCC",
            ylabel=r"Aggregated direction color accuracy (\%)",
            savename="test_pcc_direction_wo_pairs",
        )
