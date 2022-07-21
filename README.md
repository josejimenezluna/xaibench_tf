# Benchmarking molecular feature attribution methods with activity cliffs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5523952.svg)](https://doi.org/10.5281/zenodo.5523952)

Supporting data and code for "Benchmarking molecular feature attribution methods with activity cliffs", as available on [JCIM](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c01163).


## Structure of the benchmark

Download the benchmark as well as all associated data from [here](https://www.research-collection.ethz.ch/handle/20.500.11850/504716) (~85GB, when uncompressed). A smaller `benchmark.tar.gz` (~90MB) file with only the necessary files to benchmark your custom feature attribution methods (and excluding all the tested ones in manuscript) is also provided for convenience:


```bash
wget -O data.tar.gz "https://libdrive.ethz.ch/index.php/s/PYc53dCRSuxAiqC/download?path=%2F&files=data_v2.tar.gz"
tar -xf data.tar.gz
```

or alternatively:

```bash
wget -O benchmark.tar.gz "https://libdrive.ethz.ch/index.php/s/PYc53dCRSuxAiqC/download?path=%2F&files=benchmark.tar.gz"
tar -xf benchmark.tar.gz

```

The data is composed of subfolders, each contaning one congeneric series considered in the benchmark. Subfolders have the following structure:

```bash
$ tree 4ZO1-T3
4ZO1-T3
├── 4ZO1-T3.tsv
├── bench.csv
├── colors_*.pt 
├── colors.pt
├── pairs.csv
├── similarity*.npy
├── training*.csv

```

An explanation of each file is provided below:


* `4ZO1-T3.tsv`: tab-separated file containing the benchmark data as per downloaded from the BindingDB database.
* `bench.csv`: comma-separated file with all considered benchmark compounds in this series, after preprocessing, in SMILES format. 
* `colors_*.pt`: assigned colors for each of the feature attribution methods herein considered (e.g. gcn, mpnn, gat). Also includes `_wo_pairs` files that correspond to assigned atomic colors after removing common compounds from its respective training set.
* `colors.pt`: list with assigned ground-truth colors for each of the considered benchmark molecule pairs in the `pairs.csv` file. Colors were determined using different MCS common atom thresholds between pairs, from 50% to 95%, in 5% intervals. For more details on how these are computed, check the `determine_col.py` script.
* `pairs.csv`: benchmark pairs considered for each series, in SMILES format, as well as activity difference between them.
* `similarity*.npy`: numpy array containing Tanimoto similarity matrix between the compounds contained in the `training.csv` and `bench.csv` files, before and after removing benchmark compounds pairs from the training sets (see `_wo_pairs` files).
* `training*.csv`: training compounds as well as activity information for each of the targets considered in the benchmark, extracted from the ChEMBL27 database. A version excluding the benchmark compounds is provided in the `_wo_pairs` equivalents.

All the `.pt` files can be read with the Python [pickle module](https://docs.python.org/3/library/pickle.html).


## Replication of the results

All results reported in the manuscript can be reproduced with the accompanying code. We recommend the [conda](https://docs.conda.io/en/latest/miniconda.html) Python package manager, and while an GPU is technically not required to run the models and feature attribution methods reported here, it is heavily encouraged. Furthermore, the code has only been tested under Linux. Make a new environment with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate xaibench_tf
```

The `graph-attribution` repository from the Google Research team is also a requirement to run some of the scripts provided here:

```bash
git clone https://github.com/google-research/graph-attribution.git
export PYTHONPATH=/path/to/graph-attribution:$PYTHONPATH
```

An explanation of each python script is provided below for clearness:

```bash
$ tree xaibench/
xaibench/
├── agreement.py
├── color.py
├── color_utils.py
├── determine_col.py
├── diff_utils.py
├── get_training_data.py
├── make_pairs.py
├── plots
│   ├── agreement.py
│   ├── desc.py
│   ├── perf.py
│   ├── plots.py
├── retrieve_bdb_series.py
├── score.py
├── similarity.py
├── train_dnn.py
├── train_gnn.py
├── train_rf.py
└── utils.py
```

* `agreement.py`: Computes color agreement between all considered feature attribution methods considered in the benchmark.
* `color.py`: Computes molecular feature attribution colors for a specific `pairs.csv` file, using all available methods for block type `block_type` (options include `gcn, mpnn, gat, graphnet, rf, dnn`) 
* `color_utils.py`: Utility functions for `color.py`.
* `determine_col.py`: Computes ground-truth colors using MCS for a given `pairs.csv` file.
* `diff_utils.py`: Utility functions for the feature attribution method proposed by Sheridan (2019).
* `get_training_data.py`: Extracts training data from ChEMBL for all the considered series in the benchmark. Currently this assumes a working version of ChEMBL27 database on a local instance of PostgreSQL. See script for more details.
* `make_pairs.py`: Takes original benchmark data extracted from the BindingDB database, processes it (see manuscript for details), and selects benchmark pairs according to a predefined activity difference threshold.
* `retrieve_bdb_series.py`: Scrapes benchmark data from the BindingDB website.
* `score.py`: Computes color accuracy and f1 values for all feature attribution methods and benchmark pairs considered in the study.
* `similarity.py`: Computes Tanimoto similarity matrix between training and benchmark compounds.
* `train*.py`: Scripts to train the different underlying machine learning models used throughout the study. These accept a `-csv` argument to pass a file with the compounds used to train the models (i.e. the `training.csv` files).
* `utils.py`: Utility files used throughout the repo for path handling and other minor tasks. 


## (Optional) Download trained models and results

All trained models as well as generated results are also available for download. These can come in handy for result replication without executing the accompanying code, which is can be computationally expensive. These are available here.

Download trained models (`models.tar.gz`):

```bash
wget -O models.tar.gz "https://libdrive.ethz.ch/index.php/s/PYc53dCRSuxAiqC/download?path=%2F&files=models.tar.gz"
```

Logs with training metrics (`logs.tar.gz`):

```bash
wget -O logs.tar.gz "https://libdrive.ethz.ch/index.php/s/PYc53dCRSuxAiqC/download?path=%2F&files=logs.tar.gz"
```

And the final results presented in the manuscript (`results.tar.gz`):

```bash
wget -O results.tar.gz "https://libdrive.ethz.ch/index.php/s/PYc53dCRSuxAiqC/download?path=%2F&files=results_v2.tar.gz"
```

## Tutorial

If you are interested in using this work to benchmark additional feature attribution approaches, we provide a tutorial in jupyter notebook format that should cover the basics of data handling and calculation of metrics under `xaibench/notebooks/benchmark_example.ipynb`.  

## Citation

If you find this work, code, or parts thereof useful, please consider citing:

```
 @article{jimenezluna2021benchmarking,
 title={Benchmarking molecular feature attribution methods with activity cliffs},
 DOI={10.33774/chemrxiv-2021-pp88m},
 journal={ChemRxiv},
 publisher={Cambridge Open Engage},
 author={Jiménez Luna, José and Skalic, Miha and Weskamp, Nils},
 year={2021}}

```

