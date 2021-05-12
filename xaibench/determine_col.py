import argparse
import os
import dill

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from rdkit.Chem.rdFMCS import FindMCS
from tqdm import tqdm

MIN_PER_COMMON_ATOMS = np.arange(0.5, 1.0, step=0.05)
TIMEOUT_MCS = 300


def assign_colors(mol, common_idx, diff):
    """ 
    Given an rdkit mol, a set of common atom indices and an 
    activity difference value, assigns positive or negative values
    to the non-common atoms.
    """
    color = {}

    for idx in range(mol.GetNumAtoms()):
        if idx in common_idx:
            color[idx] = 0.0
        else:
            if diff > 0:
                color[idx] = 1.0
            elif diff < 0:
                color[idx] = -1.0
            else:
                color[idx] = 0.0
    return color


def color_mcs(mcs, mol_i, mol_j, diff, min_per_common):
    """ 
    Given a maximum common substructure object and a pair of rdkit mols,
    returns the assigned colors for the latter.
    """
    color_i = {}
    color_j = {}

    if not mcs.canceled and (
        min(mcs.numAtoms / mol_i.GetNumAtoms(), mcs.numAtoms / mol_j.GetNumAtoms())
        > min_per_common
    ):
        common_sub = MolFromSmarts(mcs.smartsString)
        common_atoms_idx_i = set(mol_i.GetSubstructMatch(common_sub))
        common_atoms_idx_j = set(mol_j.GetSubstructMatch(common_sub))

        color_i = assign_colors(mol_i, common_atoms_idx_i, diff)
        color_j = assign_colors(mol_j, common_atoms_idx_j, -diff)
        return color_i, color_j
    else:
        return None


def color_wrapper(pf):
    colors = []
    pair_df = pd.read_csv(pf)

    if len(pair_df) > 0:
        for smi_i, smi_j, diff in tqdm(
            zip(pair_df["smiles_i"], pair_df["smiles_j"], pair_df["diff"]),
            total=len(pair_df),
        ):
            mol_i, mol_j = MolFromSmiles(smi_i), MolFromSmiles(smi_j)
            mcs = FindMCS([mol_i, mol_j], timeout=TIMEOUT_MCS)

            res_mins = []

            for min_per_common in MIN_PER_COMMON_ATOMS:
                res = color_mcs(mcs, mol_i, mol_j, diff, min_per_common)
                res_mins.append(res)
            colors.append(res_mins)

        with open(os.path.join(os.path.dirname(pf), "colors.pt"), "wb") as handle:
            dill.dump(colors, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pairs", dest="pair_f", type=str, required=True,
    )
    args = parser.parse_args()
    color_wrapper(args.pair_f)
