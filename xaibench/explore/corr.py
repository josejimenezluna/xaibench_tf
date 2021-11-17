import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from xaibench.utils import BENCHMARK_PATH, BLOCK_TYPES

if __name__ == "__main__":
    pairs_fs = glob(os.path.join(BENCHMARK_PATH, "*", "pairs.csv"))

    corrs = {}
    corrs_wo_pairs = {}

    for pair_f in tqdm(pairs_fs):
        dirname = os.path.dirname(pair_f)
        pair_df = pd.read_csv(pair_f)

        for bt in BLOCK_TYPES + ["rf", "dnn"]:
            pred_f = os.path.join(dirname, f"preds_{bt}.npy")
            if os.path.exists(pred_f):
                preds = np.load(pred_f)
                corrs.setdefault(bt, []).append(
                    np.corrcoef(preds, pair_df["diff"])[0, 1]
                )

            pred_f_wo_pairs = os.path.join(dirname, f"preds_{bt}_wo_pairs.npy")

            if os.path.exists(pred_f_wo_pairs):
                preds_wo_pairs = np.load(pred_f_wo_pairs)
                corrs_wo_pairs.setdefault(bt, []).append(
                    np.corrcoef(preds_wo_pairs, pair_df["diff"])[0, 1]
                )

    for k, corr in corrs.items():
        print(f" {k} mean: {np.nanmean(corr):.3f}, std: {np.nanstd(corr):.3f}")


    for k, corr in corrs_wo_pairs.items():
        print(f"{k}: mean: {np.nanmean(corr):.3f}, std: {np.nanstd(corr):.3f}")
