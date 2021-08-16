import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODELS_RF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_rf")
MODELS_DNN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_dnn")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
FIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

BLOCK_TYPES = ["graphnet", "gcn", "mpnn", "gat"]


def ensure_readability(strings, read_f):
    valid_idx = []
    for idx, string in enumerate(strings):
        mol = read_f(string)
        if mol is not None:
            valid_idx.append(idx)
    return valid_idx


def translate(strings_, fromfun, tofun):
    trans = []
    idx_success = []
    for idx, s in enumerate(strings_):
        try:
            mol = fromfun(s)
            if mol is not None:
                trans.append(tofun(mol))
                idx_success.append(idx)
        except:
            continue
    return trans, idx_success
