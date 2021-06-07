import argparse
import multiprocessing
import os

import dill
import numpy as np
import pandas as pd
from joblib import dump
from rdkit.Chem import MolFromSmiles
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from xaibench.diff_utils import featurize_ecfp4
from xaibench.train_gnn import SEED, TEST_SET_SIZE, rmse
from xaibench.utils import LOG_PATH, MODELS_RF_PATH

N_TREES = 1000
N_JOBS = int(os.getenv("LSB_DJOB_NUMPROC", multiprocessing.cpu_count()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-csv", dest="csv", type=str, required=True,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    smiles, values = (
        df["canonical_smiles"].values,
        df["pchembl_value"].values,
    )

    idx_train, idx_test = train_test_split(
        np.arange(len(smiles)), random_state=SEED, test_size=TEST_SET_SIZE
    )

    smiles_train, values_train = smiles[idx_train], values[idx_train]
    smiles_test, values_test = smiles[idx_test], values[idx_test]

    fps_train = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_train])
    fps_test = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_test])

    rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=N_JOBS)
    rf.fit(fps_train, values_train)

    yhat_train = rf.predict(fps_train)
    yhat_test = rf.predict(fps_test)

    metrics = {}
    metrics["rmse_train"] = rmse(values_train, yhat_train)
    metrics["pcc_train"] = np.corrcoef((values_train, yhat_train))[0, 1]

    metrics["rmse_test"] = rmse(values_test, yhat_test)
    metrics["pcc_test"] = np.corrcoef((values_test, yhat_test))[0, 1]

    os.makedirs(MODELS_RF_PATH, exist_ok=True)
    dump(
        rf,
        os.path.join(
            MODELS_RF_PATH, f"{os.path.basename(os.path.dirname(args.csv))}.pt"
        ),
    )

    os.makedirs(LOG_PATH, exist_ok=True)
    with open(
        os.path.join(
            LOG_PATH, f"{os.path.basename(os.path.dirname(args.csv))}_metrics_rf.pt",
        ),
        "wb",
    ) as handle:
        dill.dump(metrics, handle)
