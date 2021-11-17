import os
import tarfile

from tqdm import tqdm

from xaibench.utils import BENCHMARK_PATH, ROOT_PATH

INCLUDE_F = [
    "bench.csv",
    "pairs.csv",
    "training.csv",
    "training_wo_pairs.csv",
    "colors.pt",
]

if __name__ == "__main__":
    tar_f = os.path.join(ROOT_PATH, "benchmark.tar.gz")

    with tarfile.open(tar_f, "w:gz") as tar:
        for id_ in tqdm(os.listdir(BENCHMARK_PATH)):
            dirname = os.path.join(BENCHMARK_PATH, id_)
            needed_f = [os.path.exists(os.path.join(dirname, f)) for f in INCLUDE_F]
            if all(needed_f):
                tar.add(dirname, arcname=id_, recursive=True)
