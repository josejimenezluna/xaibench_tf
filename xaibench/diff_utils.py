from copy import deepcopy

import numpy as np
import tensorflow as tf
from graph_attribution.featurization import MolTensorizer, smiles_to_graphs_tuple
from graph_attribution.graphs import get_graphs_tf, get_num_graphs
from graph_nets.utils_tf import data_dicts_to_graphs_tuple
from rdkit.Chem import AllChem, DataStructs, MolFromSmiles
from tqdm import tqdm

from xaibench.color_utils import get_batch_indices
from xaibench.train_gnn import DEVICE

FP_SIZE = 1024
BOND_RADIUS = 2


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


def featurize_ecfp4(mol, fp_size=FP_SIZE, bond_radius=BOND_RADIUS):
    """
    Gets an ECFP4 fingerprint for a specific rdkit mol. 
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def pred_pairs(pair_df, model, batch_size=16):
    tensorizer = MolTensorizer()

    g_i, g_j = (
        smiles_to_graphs_tuple(pair_df["smiles_i"], tensorizer),
        smiles_to_graphs_tuple(pair_df["smiles_j"], tensorizer),
    )
    preds_diff = []

    n = get_num_graphs(g_i)
    indices = get_batch_indices(n, int(batch_size / 2))

    for idx in tqdm(indices):
        with DEVICE:
            b_i, b_j = get_graphs_tf(g_i, idx), get_graphs_tf(g_j, idx)
            pred_i, pred_j = model(b_i), model(b_j)
            pred = pred_i - pred_j
        preds_diff.extend(pred.numpy()[:, 0].tolist())
    return preds_diff


def pred_pairs_diff(pair_df, model, mol_read_f=MolFromSmiles):
    preds_diff = []

    for row in tqdm(pair_df.itertuples(), total=len(pair_df)):
        sm_i, sm_j = getattr(row, "smiles_i"), getattr(row, "smiles_j")
        mol_i, mol_j = mol_read_f(sm_i), mol_read_f(sm_j)
        fp_i, fp_j = featurize_ecfp4(mol_i), featurize_ecfp4(mol_j)
        pred_i, pred_j = (
            model.predict(fp_i[np.newaxis, :]).squeeze(),
            model.predict(fp_j[np.newaxis, :]).squeeze(),
        )
        pred = pred_i - pred_j
        preds_diff.append(pred)
    return preds_diff


def diff_mask(
    mol_string,
    pred_fun,
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

    og_pred = pred_fun(og_fp[np.newaxis, :]).squeeze()

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)

    mod_fps = [featurize_ecfp4(mol, fp_size, bond_radius) for mol in mod_mols]
    mod_fps = np.vstack(mod_fps)
    mod_preds = pred_fun(mod_fps).squeeze()
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

