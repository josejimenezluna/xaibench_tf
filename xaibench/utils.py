import os


def ensure_readability(strings, read_f):
    valid_idx = []
    for idx, string in enumerate(strings):
        mol = read_f(string)
        if mol is not None:
            valid_idx.append(idx)
    return valid_idx


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODELS_RF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_rf")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
FIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


MODEL_PARAM = {
    "node_in_feats": 49,
    "edge_in_feats": 10,
    "n_tasks": 1,
}
