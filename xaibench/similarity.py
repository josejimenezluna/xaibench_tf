import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit.Chem import MolFromInchi, MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.DataStructs import TanimotoSimilarity

from xaibench.utils import DATA_PATH

BOND_RADIUS = 2
FP_SIZE = 1024
N_JOBS = int(os.getenv("LSB_DJOB_NUMPROC", "1"))


def tanimoto_sim(mol_i, mol_j, radius=2):
    fp_i, fp_j = (
        GetMorganFingerprint(mol_i, radius),
        GetMorganFingerprint(mol_j, radius),
    )
    return TanimotoSimilarity(fp_i, fp_j)


def parallel_wrapper(mol, rest_inchis, n_total):
    sims = np.zeros(n_total, dtype=np.float32)
    n_rest = len(rest_inchis)
    fill_idx = n_total - n_rest

    for inchi in rest_inchis:
        mol_j = MolFromInchi(inchi)
        sims[fill_idx] = tanimoto_sim(mol, mol_j)
        fill_idx += 1
    return sims


def sim_pair_train(pairs_f, training_f):
    pair_df = pd.read_csv(pairs_f)
    training_df = pd.read_csv(training_f)

    smiles_pair = []
    smiles_pair.extend(pair_df["smiles_i"].tolist())
    smiles_pair.extend(pair_df["smiles_j"].tolist())

    n_total = len(training_df)

    sims = Parallel(n_jobs=N_JOBS, verbose=100, backend="multiprocessing")(
        delayed(parallel_wrapper)(
            MolFromSmiles(smiles), training_df["standard_inchi"].tolist(), n_total
        )
        for smiles in smiles_pair
    )

    sims = np.stack(sims)
    return sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pairs", dest="pair_f", type=str, required=True,
    )
    args = parser.parse_args()

    id_ = os.path.basename(os.path.dirname(args.pair_f))
    training_f = os.path.join(DATA_PATH, "validation_sets", f"{id_}", "training.csv")

    sim_matrix = sim_pair_train(args.pair_f, training_f)
    np.save(
        os.path.join(DATA_PATH, "validation_sets", f"{id_}", "similarity.npy"),
        arr=sim_matrix,
    )