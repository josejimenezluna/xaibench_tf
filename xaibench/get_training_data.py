import os
from glob import glob

import numpy as np
import pandas as pd
import psycopg2
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToInchi, MolToSmiles, MolFromInchi
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from tqdm import tqdm

from xaibench.retrieve_bdb_series import DATA_PATH
from xaibench.utils import ensure_readability, translate
from IPython.core.debugger import Tracer; debug_here = Tracer()

RDLogger.DisableLog("rdApp.*")

UNIPROT_COL = "UniProt (SwissProt) Primary ID of Target Chain"

SQL_QUERY = """
SELECT cs.canonical_smiles, ac.standard_relation, ac.pchembl_value, ac.standard_units, ac.standard_type, ass.assay_id
FROM compound_structures AS cs INNER JOIN activities as ac ON cs.molregno = ac.molregno
INNER JOIN assays AS ass ON ac.assay_id = ass.assay_id
INNER JOIN target_components AS tc ON ass.tid = tc.tid
INNER JOIN component_sequences AS cseq ON tc.component_id = cseq.component_id
WHERE cseq.accession = '{}';
"""

TYPES = ["IC50", "Ki", "Kd"]
MIN_SAMPLES = 100


def retrieve_ligands(conn, tsv):
    """
    Queries a local ChEMBL postgres database in order to extract training data
    for a specific UniprotID from a BindingDB tsv file.
    """
    df = pd.read_csv(tsv, sep="\t")
    uniprot_ids = pd.unique(df[UNIPROT_COL])
    uniprot_ids = uniprot_ids[~pd.isna(uniprot_ids)]

    records = []

    if uniprot_ids.size > 0:
        for uniprot_id in uniprot_ids:
            cur = conn.cursor()
            cur.execute(SQL_QUERY.format(uniprot_id))
            records.extend(cur.fetchall())
            cur.close()

    records = pd.DataFrame(
        records,
        columns=[
            "canonical_smiles",
            "standard_relation",
            "pchembl_value",
            "standard_units",
            "standard_type",
            "assay_id"
        ],
    )

    records = records.loc[
        (records["standard_type"].isin(TYPES))
        & (records["standard_relation"] == "=")
        & pd.notna(records["pchembl_value"])
    ]

    valid_idx = ensure_readability(records["canonical_smiles"].to_list(), MolFromSmiles)
    training_df = records.iloc[valid_idx]
    inchis = []
    idx_suc = []

    for idx, sm in enumerate(training_df["canonical_smiles"].to_list()):
        mol = Cleanup(MolFromSmiles(sm))
        if mol is not None:
            inchis.append(MolToInchi(mol))
            idx_suc.append(idx)

    values = pd.to_numeric(training_df["pchembl_value"].iloc[idx_suc]).to_list()
    assay_ids = pd.unique(training_df["assay_id"].iloc[idx_suc].to_list())

    df_inchis = pd.DataFrame({"inchis": inchis, "pchembl_value": values})
    records = df_inchis.groupby(["inchis"], as_index=False)[
        "pchembl_value"
    ].mean()

    smiles, idx_trans = translate(records["inchis"], MolFromInchi, MolToSmiles)
    values = records["pchembl_value"].iloc[idx_trans].to_list()

    train_clean = pd.DataFrame(
        {"canonical_smiles": smiles, "pchembl_value": values}
    )
    return train_clean, assay_ids


if __name__ == "__main__":
    conn = psycopg2.connect("dbname=chembl_27 user=hawk31")
    tsvs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "*.tsv"))

    for tsv in tqdm(tsvs):
        dirname = os.path.dirname(tsv)
        bench_csv = os.path.join(dirname, "bench.csv")
        colors_pt = os.path.join(dirname, "colors.pt")
        bench_df = pd.read_csv(bench_csv)

        if os.path.exists(bench_csv) and os.path.exists(colors_pt):
            df, assay_ids = retrieve_ligands(conn, tsv)
            if len(df) > MIN_SAMPLES:
                df.to_csv(os.path.join(dirname, "training.csv"), index=None)
                np.save(os.path.join(dirname, "assay_ids.npy"), arr=assay_ids)

                bench_sm = set(bench_df["smiles"].to_list())
                idx_notinbench = []
                for idx, sm_train in enumerate(
                    df["canonical_smiles"].to_list()
                ):
                    if sm_train not in bench_sm:
                        idx_notinbench.append(idx)

                train_wo = df.iloc[idx_notinbench]
                if len(train_wo) > MIN_SAMPLES:
                    train_wo.to_csv(
                        os.path.join(dirname, "training_wo_pairs.csv"), index=None
                    )

