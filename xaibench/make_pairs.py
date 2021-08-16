import os

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToInchi, MolFromInchi, MolToSmiles
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from tqdm import tqdm

from xaibench.retrieve_bdb_series import SET_PATH
from xaibench.utils import ensure_readability, translate

MW_THR = 800
MIN_P_DIFF = 1.0
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
    Looks for activity cliffs in the protein-ligand validation sets on BindingDB,
    filtering by a log activity factor difference of min_p_diff and a molecular
    weight factor of mw_thr
    """
    df = pd.read_csv(tsv_file, sep="\t")
    df = df.loc[df[affcol].notna()]

    if df[affcol].dtype == np.dtype("O"):
        df = df[~df[affcol].str.contains("|".join(SYMBOL_EXCLUDE))]

    ok_read_idx = ensure_readability(df[ligcol].to_list(), MolFromSmiles)

    df = df.iloc[ok_read_idx]
    values = df[affcol].values.astype(np.float32)

    inchis = []
    idx_suc = []
    for idx, sm in enumerate(df[ligcol]):
        mol = Cleanup(MolFromSmiles(sm))
        if mol is not None:
            inchis.append(MolToInchi(mol))
            idx_suc.append(idx)

    values = values[idx_suc]

    df_clean = pd.DataFrame({"inchis": inchis, "values": values})
    df_clean = df_clean.groupby(["inchis"], as_index=False)["values"].mean()

    smiles, idx_trans = translate(df_clean["inchis"], MolFromInchi, MolToSmiles)
    values = df_clean["values"].iloc[idx_trans].values
    smiles = np.array(smiles)

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
            df_filtered = pd.DataFrame({"smiles": pairs_df["smiles_i"].to_list() + pairs_df["smiles_j"].to_list()})
            df_filtered.drop_duplicates(inplace=True)
            df_filtered.to_csv(os.path.join(os.path.dirname(tsv_file), "bench.csv"), index=None)

            pairs_df.to_csv(
                os.path.join(os.path.dirname(tsv_file), "pairs.csv"), index=None
            )


if __name__ == "__main__":
    ids = os.listdir(SET_PATH)

    for id_ in tqdm(ids):
        process_tsv(os.path.join(SET_PATH, id_, "{}.tsv".format(id_)))
