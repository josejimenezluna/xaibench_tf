import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from xaibench.utils import DATA_PATH


def check_offenders(pairs_df):
    """
    Checks whether there are ligands with opposite colors depending
    on the benchmarking pair they are in.
    """
    un_ligands = np.unique(
        [pairs_df["smiles_i"].to_list() + pairs_df["smiles_j"].to_list()]
    )
    offenders = 0

    for lig in un_ligands:
        df_subset = pairs_df.loc[
            (pairs_df["smiles_i"] == lig) | (pairs_df["smiles_j"] == lig)
        ]
        df_subset_i = pairs_df.loc[(pairs_df["smiles_i"] == lig)]
        df_subset_j = pairs_df.loc[(pairs_df["smiles_j"] == lig)]

        # flip sign of diff on df_j and check sign is the same
        df_subset_j.loc[:, "diff"] = df_subset_j["diff"].apply(lambda x: -x)

        if len(df_subset) >= 2:
            diffs = df_subset_i["diff"].to_list() + df_subset_j["diff"].to_list()
            sign = np.sign(diffs)
            if np.all(sign > 0) or np.all(sign < 0):
                pass  # all good
            else:
                offenders += 1
    return offenders, len(un_ligands)


if __name__ == "__main__":
    pairs_fs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "pairs.csv"))
    n_offenders, n_total = [], []

    for pair_f in tqdm(pairs_fs):
        pairs_df = pd.read_csv(pair_f)
        n_o, n_t = check_offenders(pairs_df)
        n_offenders.append(n_o)
        n_total.append(n_t)

    ratio = [(n_o / n_t) for n_o, n_t in zip(n_offenders, n_total)]
    print(sum(n_offenders) / sum(n_total)) # 54.42 %
    print(np.mean(ratio)) # 32 %
