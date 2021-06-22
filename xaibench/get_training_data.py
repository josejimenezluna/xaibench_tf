import os
from glob import glob

import numpy as np
import pandas as pd
import psycopg2
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToInchi
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from tqdm import tqdm

from xaibench.retrieve_bdb_series import DATA_PATH
from xaibench.utils import ensure_readability

RDLogger.DisableLog("rdApp.*")

UNIPROT_COL = "UniProt (SwissProt) Primary ID of Target Chain"

SQL_QUERY = """
SELECT cs.canonical_smiles, ac.standard_relation, ac.pchembl_value, ac.standard_units, ac.standard_type
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
        ],
    )

    records = records.loc[
        (records["standard_type"].isin(TYPES))
        & (records["standard_relation"] == "=")
        & pd.notna(records["pchembl_value"])
    ]

    valid_idx = ensure_readability(records["canonical_smiles"].to_list(), MolFromSmiles)
    records = records.iloc[valid_idx]
    return records[["canonical_smiles", "pchembl_value"]]


if __name__ == "__main__":
    conn = psycopg2.connect("dbname=chembl_27 user=hawk31")
    tsvs = glob(os.path.join(DATA_PATH, "validation_sets", "*", "*.tsv"))

    for tsv in tqdm(tsvs):
        dirname = os.path.dirname(tsv)
        df = retrieve_ligands(conn, tsv)
        if len(df) > MIN_SAMPLES:
            df.to_csv(os.path.join(dirname, "training.csv"), index=None)

        bench_csv = os.path.join(dirname, "bench.csv")

        # remove training ligands if present in benchmark series

        if os.path.exists(bench_csv):
            inchis_training = [
                MolToInchi(Cleanup(MolFromSmiles(sm))) for sm in df["canonical_smiles"]
            ]
            id_ = os.path.basename(dirname)
            bench_df = pd.read_csv(bench_csv)

            inchis_bench = set(
                [MolToInchi(Cleanup(MolFromSmiles(sm))) for sm in bench_df["smiles"]]
            )

            noncommon_idx = []
            for idx, inchi_t in enumerate(inchis_training):
                if inchi_t not in inchis_bench:
                    noncommon_idx.append(idx)

            df_filtered = df.iloc[noncommon_idx]
            if len(df_filtered) > MIN_SAMPLES:
                df_filtered.to_csv(
                    os.path.join(dirname, "training_wo_pairs.csv"), index=None
                )

