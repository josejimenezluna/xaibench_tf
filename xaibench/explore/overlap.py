import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from xaibench.utils import DATA_PATH, RESULTS_PATH

if __name__ == "__main__":
    bench_csvs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "bench.csv"))

    percentages = []

    for bench_csv in tqdm(bench_csvs):
        dirname = os.path.dirname(bench_csv)
        train_csv = os.path.join(dirname, "training.csv")
        train_wo_csv= os.path.join(dirname, "training_wo_pairs.csv")

        if os.path.exists(train_csv) and os.path.exists(train_wo_csv):
            bench_df = pd.read_csv(bench_csv)
            train_df = pd.read_csv(train_csv)
            train_wo_df = pd.read_csv(train_wo_csv)

            percentages.append((len(train_df) - len(train_wo_df)) / len(bench_df))

    np.save(os.path.join(RESULTS_PATH, "overlap.npy"), arr=percentages)
