import os
from glob import glob
from tqdm import tqdm 
import numpy as np

from xaibench.utils import DATA_PATH, FIG_PATH
import matplotlib.pyplot as plt

if __name__ == "__main__":
    similarity_fs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "similarity.npy"))
    avg_similarities = []
    max_similarities = []

    for sim_f in tqdm(similarity_fs):
        sim = np.load(sim_f)
        avg_similarities.extend(sim.mean(axis=1))
        max_similarities.extend(sim.max(axis=1))

    plt.hist(avg_similarities, bins=50, label="average")
    plt.hist(max_similarities, bins=50, label="max")
    plt.axvline(np.median(avg_similarities), linestyle="--", color="black")
    plt.axvline(np.median(max_similarities), linestyle="--", color="black")
    plt.title("Tanimoto similarity between training and test sets")
    plt.legend(loc="upper center")
    plt.savefig(os.path.join(FIG_PATH, "train_test_sim.png"))
    plt.close()
