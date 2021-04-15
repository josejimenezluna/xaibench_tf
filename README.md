# xaibench_tf

Conda python installation required. A GPU is encouraged. Replicate the provided environment with:

```bash
conda env create -f environment.yml
```

The `graph-attribution` repository from Google Research is also a requirement to run some of the scripts provided here:

```bash
git clone https://github.com/google-research/graph-attribution.git
export PYTHONPATH=/path/to/graph-attribution:$PYTHONPATH
```

Since you will be training models internally, I suggest you first find a task for which you have enough data (e.g. in the order of the thousand of molecules). Once that's decided:

1. A `pairs.csv` file needs to be created with pairs of compounds in the SMILES format for which a certain property difference exists. To do this, take the `make_pairs.py` script as a reference. The latter script needs to be adapted to your use case as it is written with the compounds from BindingDB in mind. This is the only part where custom code needs to be written on your side.

2. Then, the maximum common substructures between the proposed pairs needs to be computed, and then filtered out accordingly. Check the `determine_col.py` script in order to do this. Note: MCS calculations can take quite some time depending on the pair, and so, a `TIMEOUT` variable was set. If there are many pairs for which MCSs need to be computed, consider restricting the set of pairs to a smaller one as you consider fit.

3. Both graph-neural-network and random-forest models need to be trained. For the former 3 different block types are available (GCN, MPPN, GAT), check `train_gnn.py` for more details. The random forest models can be trained via the `train_rf.py` script. 

4. Molecules can be then colored using the trained models, with the `color.py`. As in step 3, the script needs to be run with different block types.

5. Plots and final analyses can be consulted with the `check.py` script. Other scripts, such as `train_test_sim.py` are needed to be run in order for some plots to be produced (namely the one concerning the influence of train/test similarity on the colorings).
