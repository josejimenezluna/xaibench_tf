from copy import deepcopy

import numpy as np
import tensorflow as tf
from graph_attribution.featurization import MolTensorizer
from graph_nets.utils_tf import data_dicts_to_graphs_tuple
from rdkit.Chem import AllChem, DataStructs, MolFromSmiles

from xaibench.train_gnn import DEVICE


def gen_dummy_atoms(mol, dummy_atom_no=47):
    """
    Given a specific rdkit mol, returns a list of mols where each individual atom
    has been replaced by a dummy atom type.
    """
    mod_mols = []

    for idx_atom in range(mol.GetNumAtoms()):
        mol_cpy = deepcopy(mol)
        mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
        mod_mols.append(mol_cpy)
    return mod_mols


def featurize_ecfp4(mol, fp_size=1024, bond_radius=2):
    """
    Gets an ECFP4 fingerprint for a specific rdkit mol. 
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def diff_rf(
    mol_string,
    model,
    task="regression",
    fp_size=1024,
    bond_radius=2,
    dummy_atom_no=47,
    mol_read_f=MolFromSmiles,
):
    """
    Given a mol specified by a string (SMILES, inchi), uses Sheridan's method (2019)
    alongside an sklearn model to compute atom attribution.
    """
    mol = mol_read_f(mol_string)
    og_fp = featurize_ecfp4(mol, fp_size, bond_radius)

    pred_fun = (
        lambda x: model.predict_proba(x)[:, 1]
        if task == "binary"
        else lambda x: model.predict(x)
    )

    og_pred = pred_fun(og_fp[np.newaxis, :])

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)

    mod_fps = [featurize_ecfp4(mol, fp_size, bond_radius) for mol in mod_mols]
    mod_fps = np.vstack(mod_fps)
    mod_preds = pred_fun(mod_fps)
    return og_pred - mod_preds


def gen_masked_atom_feats(og_g):
    """ 
    Given a graph, returns a list of graphs where individual atoms
    are masked.
    """
    masked_gs = []
    for node_idx in range(og_g[0]["nodes"].shape[0]):
        g = deepcopy(og_g)
        g[0]["nodes"][node_idx] *= 0.0
        masked_gs.append(g[0])
    return masked_gs


def diff_gnn(smiles, model):
    """ 
    Given a SMILES string, uses Sheridan's method (2019) alongside
    a trained GNN model to compute atom attribution.
    """
    tensorizer = MolTensorizer()

    og_g = tensorizer.transform_data_dict([smiles])
    masked_gs = gen_masked_atom_feats(og_g)

    og_gt = data_dicts_to_graphs_tuple(og_g)
    gts = data_dicts_to_graphs_tuple(masked_gs)
    with DEVICE:
        og_pred = model(og_gt)
        mod_preds = model(gts)
    return tf.squeeze(og_pred - mod_preds).numpy()

