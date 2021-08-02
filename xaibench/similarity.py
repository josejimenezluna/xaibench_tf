import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.DataStructs import TanimotoSimilarity

from xaibench.utils import DATA_PATH

BOND_RADIUS = 2
FP_SIZE = 1024
N_JOBS = int(os.getenv("LSB_DJOB_NUMPROC", multiprocessing.cpu_count()))


def tanimoto_sim(mol_i, mol_j, radius=2):
    """ 
    Returns tanimoto similarity for a pair of mols
    """
    fp_i, fp_j = (
        GetMorganFingerprint(mol_i, radius),
        GetMorganFingerprint(mol_j, radius),
    )
    return TanimotoSimilarity(fp_i, fp_j)


def parallel_wrapper(mol, rest_smiles, n_total):
    """ 
    Wrapper for similarity computation over the rows of the matrix.
    """
    sims = np.zeros(n_total, dtype=np.float32)
    n_rest = len(rest_smiles)
    fill_idx = n_total - n_rest

    for smiles in rest_smiles:
        mol_j = MolFromSmiles(smiles)
        sims[fill_idx] = tanimoto_sim(mol, mol_j)
        fill_idx += 1
    return sims


def sim_pair_train(pairs_f, training_f):
    """ 
    Computes train/test tanimoto similarity matrices for the compounds in
    BindingDB and the ChEMBL database. 
    """
    pair_df = pd.read_csv(pairs_f)
    training_df = pd.read_csv(training_f)

    smiles_pair = []
    smiles_pair.extend(pair_df["smiles_i"].tolist())
    smiles_pair.extend(pair_df["smiles_j"].tolist())

    n_total = len(training_df)

    sims = Parallel(n_jobs=N_JOBS, verbose=100, backend="multiprocessing")(
        delayed(parallel_wrapper)(
            MolFromSmiles(smiles), training_df["canonical_smiles"].tolist(), n_total
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
    training_f_wo = os.path.join(DATA_PATH, "validation_sets", f"{id_}", "training_wo_pairs.csv")

    if os.path.exists(training_f):
        sim = sim_pair_train(args.pair_f, training_f)
        np.save(
            os.path.join(DATA_PATH, "validation_sets", f"{id_}", "similarity.npy"),
            arr=sim,
        )
    
    if os.path.exists(training_f_wo):
        sim_wo = sim_pair_train(args.pair_f, training_f_wo)
        np.save(
            os.path.join(DATA_PATH, "validation_sets", f"{id_}", "similarity_wo_pairs.npy"),
            arr=sim_wo,
        )
