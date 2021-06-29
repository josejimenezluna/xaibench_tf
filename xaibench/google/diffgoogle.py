import argparse
import collections
import os

import dill
import numpy as np
import pandas as pd
import sonnet as snt
from graph_attribution.experiments import GNN
from graph_attribution.featurization import (MolTensorizer,
                                             smiles_to_graphs_tuple)
from graph_attribution.graphnet_models import BlockType
from graph_attribution.hparams import get_hparams
from graph_attribution.tasks import BinaryClassificationTaskType
from graph_attribution.templates import TargetType
from graph_attribution.training import make_tf_opt_epoch_fn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from xaibench.google.rfgoogle import TEST_SUITES
from xaibench.google.utils import GADATA_PATH, RES_PATH
from xaibench.train_gnn import (BATCH_SIZE, DEVICE, HID_SIZE, LR, N_EPOCHS,
                                N_LAYERS)
from xaibench.diff_utils import diff_gnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-block_type", dest="block_type", type=str, required=False, default="gcn"
    )
    args = parser.parse_args()

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
        smiles_train, smiles_test = (
            df["smiles"].values[train_idxs],
            df["smiles"].values[test_idxs],
        )
        values_train, values_test = (
            df["label"].values[train_idxs][:, np.newaxis],
            df["label"].values[test_idxs][:, np.newaxis],
        )

        tensorizer = MolTensorizer()
        graph_train = smiles_to_graphs_tuple(smiles_train, tensorizer)
        graph_test = smiles_to_graphs_tuple(smiles_test, tensorizer)

        hp = get_hparams(
            {
                "node_size": HID_SIZE,
                "edge_size": HID_SIZE,
                "global_size": HID_SIZE,
                "block_type": args.block_type,
                "learning_rate": LR,
                "epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "n_layers": N_LAYERS,
            }
        )

        task_act = BinaryClassificationTaskType().get_nn_activation_fn()
        task_loss = BinaryClassificationTaskType().get_nn_loss_fn()
        target_type = TargetType("globals")

        with DEVICE:
            model = GNN(
                node_size=hp.node_size,
                edge_size=hp.edge_size,
                global_size=hp.global_size,
                y_output_size=1,
                block_type=BlockType(hp.block_type),
                activation=task_act,
                target_type=target_type,
                n_layers=hp.n_layers,
            )
            model(graph_train)  # one pass needed for init

            optimizer = snt.optimizers.Adam(hp.learning_rate)

            opt_one_epoch = make_tf_opt_epoch_fn(
                graph_train,
                values_train,
                hp.batch_size,
                model,
                optimizer,
                task_loss,
                l2_reg=0.0,
            )

            pbar = tqdm(range(hp.epochs))
            metrics = collections.defaultdict(list)

            for _ in pbar:
                train_loss = opt_one_epoch(graph_train, values_train).numpy()
                y_hat_train = model(graph_train).numpy().squeeze()
                y_hat_test = model(graph_test).numpy().squeeze()

                metrics["auc_train"].append(roc_auc_score(values_train.squeeze(), y_hat_train))
                metrics["auc_test"].append(roc_auc_score(values_test.squeeze(), y_hat_test))
                pbar.set_postfix({key: values[-1] for key, values in metrics.items()})
        
        aucs[t_suite] = metrics["auc_test"][-1]
        
        # Generate attributions
        att_pred = []
        for sm_test in tqdm(smiles_test):
            att_pred.append(
                diff_gnn(sm_test, model)
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
    with open(os.path.join(RES_PATH, f"results_{args.block_type}.pt"), "wb") as handle:
        dill.dump([aucs, att_aucs], handle)