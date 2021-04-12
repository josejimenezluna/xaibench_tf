import numpy as np
from rdkit.Chem import AllChem, DataStructs
from copy import deepcopy

GREEN_COL = (0, 1, 0)
RED_COL = (1, 0, 0)


def determine_atom_col(atom_importance, eps=1e-5):
    atom_col = {}

    for idx, v in enumerate(atom_importance):
        if v > eps:
            atom_col[idx] = GREEN_COL
        if v < -eps:
            atom_col[idx] = RED_COL
    return atom_col


def determine_bond_col(atom_col, mol):
    bond_col = {}

    for idx, bond in enumerate(mol.GetBonds()):
        atom_i_idx, atom_j_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom_i_idx in atom_col and atom_j_idx in atom_col:
            if atom_col[atom_i_idx] == atom_col[atom_j_idx]:
                bond_col[idx] = atom_col[atom_i_idx]
    return bond_col


def gen_dummy_atoms(mol, dummy_atom_no=47):
    mod_mols = []
    for idx_atom in range(mol.GetNumAtoms()):
        mol_cpy = deepcopy(mol)
        mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
        mod_mols.append(mol_cpy)
    return mod_mols


def featurize_ecfp4(mol, fp_size=1024, bond_radius=2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def diff_importance(
    mol, model, task="regression", fp_size=1024, bond_radius=2, dummy_atom_no=47
):
    og_fp = featurize_ecfp4(mol, fp_size, bond_radius)

    if task == "regression":
        pred_fun = lambda x: model.predict(x)
    elif task == "binary":
        pred_fun = lambda x: model.predict_proba(x)[:, 1]

    og_pred = pred_fun(og_fp[np.newaxis, :])

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)

    mod_fps = [featurize_ecfp4(mol) for mol in mod_mols]
    mod_fps = np.vstack(mod_fps)
    mod_preds = pred_fun(mod_fps)
    return og_pred - mod_preds

