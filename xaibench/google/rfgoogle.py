import os
import dill

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from xaibench.diff_utils import diff_mask, featurize_ecfp4
from xaibench.google.utils import GADATA_PATH, RES_PATH


TEST_SUITES = ["logic7", "logic8", "logic10", "benzene"]

if __name__ == "__main__":
    aucs = {}
    att_aucs = {}

    for t_suite in TEST_SUITES:
        print("Now evaluating suite {}...".format(t_suite))
        df = pd.read_csv(
            os.path.join(GADATA_PATH, f"{t_suite}", f"{t_suite}_smiles.csv")
        )
        idxs = np.load(
            os.path.join(GADATA_PATH, f"{t_suite}", f"{t_suite}_traintest_indices.npz")
        )
        train_idxs, test_idxs = idxs["train_index"], idxs["test_index"]

        fps = []

        for sm in tqdm(df["smiles"]):
            mol = MolFromSmiles(sm)
            fps.append(featurize_ecfp4(mol))

        fps = np.vstack(fps)
        fps_train, fps_test = fps[train_idxs, :], fps[test_idxs, :]
        label_train, label_test = (
            df["label"].values[train_idxs],
            df["label"].values[test_idxs],
        )

        rf = RandomForestClassifier(n_jobs=-1, n_estimators=10000)
        rf.fit(fps_train, label_train)

        pred_test = rf.predict_proba(fps_test)[:, 1]
        aucs[t_suite] = roc_auc_score(label_test, pred_test)

        # Generate attributions
        smiles_test = df["smiles"].iloc[test_idxs].to_list()

        att_pred = []

        for sm_test in tqdm(smiles_test):
            att_pred.append(
                diff_mask(sm_test, pred_fun=lambda x: rf.predict_proba(x)[:, 1])
            )

        att_pred = np.array(att_pred)

        # test attributions
        allatt = np.load(
            os.path.join(
                GADATA_PATH, f"{t_suite}", "true_raw_attribution_datadicts.npz"
            ),
            allow_pickle=True,
        )["datadict_list"]
        att_true = []

        for elem in tqdm(allatt):
            att_true.append(elem[0]["nodes"])

        ## TODO: figure out logic with several ground truths
        att_true = np.array(att_true)[test_idxs]
        weird_idxs = []
        for idx, t_att in enumerate(tqdm(att_true)):
            if t_att.shape[1] != 1:
                weird_idxs.append(idx)

        print("weird idxs {}/{}".format(len(weird_idxs), len(att_true)))

        non_weird_idxs = np.setdiff1d(np.arange(len(att_true)), weird_idxs)
        att_true = att_true[non_weird_idxs]
        att_pred = att_pred[non_weird_idxs]

        att_true = np.array([item[0] for sublist in att_true for item in sublist])
        att_pred = np.array([item for sublist in att_pred for item in sublist])

        att_aucs[t_suite] = roc_auc_score(att_true, att_pred)

    os.makedirs(RES_PATH, exist_ok=True)
    with open(os.path.join(RES_PATH, "results_rf.pt"), "wb") as handle:
        dill.dump([aucs, att_aucs], handle)
