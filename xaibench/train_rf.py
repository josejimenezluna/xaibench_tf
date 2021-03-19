import argparse
import os
import pickle

import numpy as np
import pandas as pd
from joblib import dump
from rdkit.Chem import AllChem, DataStructs, MolFromInchi
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from xaibench.utils import LOG_PATH, MODELS_RF_PATH


def featurize_ecfp4(mol, fp_size=1024, bond_radius=2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



TEST_SEED = 1337
N_TREES = 1000
N_JOBS = int(os.environ["LSB_DJOB_NUMPROC"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-csv", dest="csv_path", type=str, required=True,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    inchis, values = (
        df["standard_inchi"].values,
        df["pchembl_value"].values,
    )

    idx_train, idx_test = train_test_split(
        np.arange(len(inchis)), test_size=0.2, random_state=TEST_SEED
    )

    fps = np.vstack([featurize_ecfp4(MolFromInchi(inchi)) for inchi in inchis])

    fps_train, fps_test = fps[idx_train, :], fps[idx_test, :]
    values_train, values_test = values[idx_train], values[idx_test]

    rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=N_JOBS)
    rf.fit(fps_train, values_train)
    yhat_test = rf.predict(fps_test)

    r = np.corrcoef((values_test, yhat_test))[0, 1]
    rmse = np.sqrt(np.mean((values_test - yhat_test) ** 2))

    os.makedirs(MODELS_RF_PATH, exist_ok=True)
    dump(
        rf,
        os.path.join(
            MODELS_RF_PATH, f"{os.path.basename(os.path.dirname(args.csv_path))}.pt"
        ),
    )

    with open(
        os.path.join(
            LOG_PATH,
            f"{os.path.basename(os.path.dirname(args.csv_path))}_metrics_rf.pt",
        ),
        "wb",
    ) as handle:
        pickle.dump([rmse, r], handle)
