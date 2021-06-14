import os
from glob import glob

import dill
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToInchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import rdBase
from tqdm import tqdm
from xaibench.utils import DATA_PATH, RESULTS_PATH

rdBase.DisableLog("rdApp.*")

if __name__ == "__main__":
    train_csvs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "training.csv"))
    ids_w_train_and_pairs = []
    ids = os.listdir(os.path.join(DATA_PATH, "validation_sets"))

    for id_ in ids:
        if os.path.exists(
            os.path.join(DATA_PATH, "validation_sets", id_, "training.csv")
        ) and os.path.exists(
            os.path.join(DATA_PATH, "validation_sets", id_, "pairs.csv")
        ):
            ids_w_train_and_pairs.append(id_)

    common_perc = {}

    for id_ in tqdm(ids_w_train_and_pairs):
        training_df = pd.read_csv(
            os.path.join(DATA_PATH, "validation_sets", id_, "training.csv")
        )
        pairs_df = pd.read_csv(
            os.path.join(DATA_PATH, "validation_sets", id_, "pairs.csv")
        )
        pairs = []
        pairs.extend(pairs_df["smiles_i"])
        pairs.extend(pairs_df["smiles_j"])
        pairs = np.unique(pairs)

        inchis_std_train = set(
            [
                MolToInchi(rdMolStandardize.Cleanup(MolFromSmiles(sm)))
                for sm in training_df["canonical_smiles"]
            ]
        )
        inchis_std_pairs = [
            MolToInchi(rdMolStandardize.Cleanup(MolFromSmiles(sm))) for sm in pairs
        ]

        count_common = 0
        for inchi_pair in inchis_std_pairs:
            if inchi_pair in inchis_std_train:
                count_common += 1
        common_perc[id_] = count_common / len(inchis_std_pairs)

    with open(os.path.join(RESULTS_PATH, "train_pairs_overlap.pt", "wb")) as handle:
        dill.dump(common_perc, handle)
