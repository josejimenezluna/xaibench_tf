import os

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
from tqdm import tqdm

from xaibench.retrieve_bdb_series import SET_PATH
from xaibench.utils import ensure_readability

MW_THR = 800
MIN_P_DIFF = 2.0
SYMBOL_EXCLUDE = set(["=", ">", "<"])


def process_tsv(
    tsv_file,
    ligcol="Ligand SMILES",
    affcol="IC50 (nM)",
    use_log=True,
    min_p_diff=MIN_P_DIFF,
    mw_thr=MW_THR,
):
    """
    Looks activity cliffs for the protein-ligand validation sets on BindingDB,
    using a log activity factor difference of min_p_diff and a molecular
    weight factor of mw_thr
    """
    df = pd.read_csv(tsv_file, sep="\t")
    df = df.loc[df[affcol].notna()]

    if df[affcol].dtype == np.dtype("O"):
        df = df[~df[affcol].str.contains("|".join(SYMBOL_EXCLUDE))]

    ok_read_idx = ensure_readability(df[ligcol].to_list(), MolFromSmiles)

    df = df.iloc[ok_read_idx]

    smiles = df[ligcol].values
    values = df[affcol].values.astype(np.float32)

    mws = np.array([MolWt(MolFromSmiles(lig)) for lig in smiles])
    idx_below_mw = np.argwhere(mws <= mw_thr).flatten()

    smiles = smiles[idx_below_mw]
    values = values[idx_below_mw]

    if use_log:
        values = -np.log10(1e-9 * values)

    if len(smiles) >= 2:
        diff_mat = np.subtract.outer(values, values)
        diff_mat[
            np.tril_indices_from(diff_mat)
        ] = 0.0  # difference always in this direction
        idx_diff = np.argwhere(np.abs(diff_mat) > min_p_diff)

        if len(idx_diff) > 0:
            pairs_df = pd.DataFrame(
                {
                    "smiles_i": [smiles[idx[0]] for idx in idx_diff],
                    "smiles_j": [smiles[idx[1]] for idx in idx_diff],
                    "diff": [diff_mat[tuple(idx)] for idx in idx_diff],
                }
            )

            if len(pairs_df) > 0:
                pairs_df.to_csv(
                    os.path.join(os.path.dirname(tsv_file), "pairs.csv"), index=None
                )


if __name__ == "__main__":
    ids = os.listdir(SET_PATH)

    for id_ in tqdm(ids):
        process_tsv(os.path.join(SET_PATH, id_, "{}.tsv".format(id_)))
