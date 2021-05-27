import argparse
import collections
import os
from contextlib import nullcontext

import dill
import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from graph_attribution.experiments import GNN
from graph_attribution.featurization import MolTensorizer, smiles_to_graphs_tuple
from graph_attribution.graphnet_models import BlockType
from graph_attribution.hparams import get_hparams
from graph_attribution.tasks import RegresionTaskType
from graph_attribution.templates import TargetType
from graph_attribution.training import make_tf_opt_epoch_fn
from tqdm import tqdm

from xaibench.utils import LOG_PATH, MODELS_PATH

GPUS = tf.config.list_physical_devices("GPU")
N_EPOCHS = 1000
LR = 1e-3
HID_SIZE = 128
N_LAYERS = 3
BATCH_SIZE = 64

if GPUS:
    tf.config.experimental.set_memory_growth(GPUS[0], True)
    DEVICE = tf.device("/GPU:0")
else:
    DEVICE = nullcontext()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-csv", dest="csv", type=str, required=True,
    )
    parser.add_argument(
        "-block_type", dest="block_type", type=str, required=False, default="gcn"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    id_ = os.path.basename(os.path.dirname(args.csv))

    smiles, values = (
        df["canonical_smiles"].values,
        df["pchembl_value"].values,
    )

    tensorizer = MolTensorizer()
    graph_data = smiles_to_graphs_tuple(smiles, tensorizer)

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
    task_act = RegresionTaskType().get_nn_activation_fn()
    task_loss = RegresionTaskType().get_nn_loss_fn()
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
        model(graph_data)  # one pass needed for init

        optimizer = snt.optimizers.Adam(hp.learning_rate)

        opt_one_epoch = make_tf_opt_epoch_fn(
            graph_data, values, hp.batch_size, model, optimizer, task_loss, l2_reg=1.0
        )

        pbar = tqdm(range(hp.epochs))
        metrics = collections.defaultdict(list)


        for _ in pbar:
            train_loss = opt_one_epoch(graph_data, values).numpy()
            y_hat = model(graph_data).numpy().squeeze()
            metrics["rmse_train"].append(np.sqrt(train_loss))
            metrics["rs_train"].append(np.corrcoef(y_hat, values)[0, 1])

            pbar.set_postfix({key: values[-1] for key, values in metrics.items()})

    os.makedirs(os.path.join(MODELS_PATH, f"{args.block_type}_{id_}"), exist_ok=True)

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.save(
        os.path.join(
            MODELS_PATH, f"{args.block_type}_{id_}", f"{args.block_type}_{id_}"
        )
    )

    os.makedirs(LOG_PATH, exist_ok=True)

    with open(os.path.join(LOG_PATH, f"{args.block_type}_{id_}.pt"), "wb") as handle:
        dill.dump(metrics, handle)
