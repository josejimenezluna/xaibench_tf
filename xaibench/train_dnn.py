import argparse
import multiprocessing
import os

import dill
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from xaibench.diff_utils import featurize_ecfp4, FP_SIZE
from xaibench.train_gnn import BATCH_SIZE, N_EPOCHS, SEED, TEST_SET_SIZE, rmse
from xaibench.utils import LOG_PATH, MODELS_DNN_PATH

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

    model = Sequential()
    model.add(Dense(512, input_shape=(FP_SIZE,), activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    model.fit(
        fps_train, values_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, workers=N_JOBS
    )

    yhat_train = model.predict(fps_train).squeeze()
    yhat_test = model.predict(fps_test).squeeze()

    metrics = {}
    metrics["rmse_train"] = rmse(values_train, yhat_train)
    metrics["pcc_train"] = np.corrcoef((values_train, yhat_train))[0, 1]

    metrics["rmse_test"] = rmse(values_test, yhat_test)
    metrics["pcc_test"] = np.corrcoef((values_test, yhat_test))[0, 1]

    os.makedirs(MODELS_DNN_PATH, exist_ok=True)
    model.save(
        os.path.join(
            MODELS_DNN_PATH, f"{os.path.basename(os.path.dirname(args.csv))}.pt"
        )
    )

    os.makedirs(LOG_PATH, exist_ok=True)
    with open(
        os.path.join(
            LOG_PATH, f"{os.path.basename(os.path.dirname(args.csv))}_metrics_dnn.pt",
        ),
        "wb",
    ) as handle:
        dill.dump(metrics, handle)
