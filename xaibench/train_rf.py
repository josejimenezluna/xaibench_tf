import argparse
import multiprocessing
import os
import dill

import numpy as np
import pandas as pd
from joblib import dump
from rdkit.Chem import MolFromSmiles
from sklearn.ensemble import RandomForestRegressor

from xaibench.diff_utils import featurize_ecfp4
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

    fps = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles])

    rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=N_JOBS)
    rf.fit(fps, values)
    yhat = rf.predict(fps)

    r = np.corrcoef((values, yhat))[0, 1]
    rmse = np.sqrt(np.mean((values - yhat) ** 2))

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
            LOG_PATH,
            f"{os.path.basename(os.path.dirname(args.csv))}_metrics_rf.pt",
        ),
        "wb",
    ) as handle:
        dill.dump([rmse, r], handle)
