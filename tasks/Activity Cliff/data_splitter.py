### Modified from https://github.com/molML/MoleculeACE

import json
import math
import random
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from rdkit import Chem
from rdkit import rdBase
from itertools import combinations
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from Levenshtein import distance as levenshtein

warnings.filterwarnings("ignore", category=RuntimeWarning)
rdBase.DisableLog('rdApp.*')  # Disable RDKit logs


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# Similarity Matrix Calculation Functions
def get_levenshtein_matrix(smiles: List[str], normalize: bool = True, hide: bool = False, top_n: int = None):
    """Calculates a Levenshtein similarity matrix for a list of SMILES strings."""

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])

    # Compute the upper triangle of the matrix
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            if normalize:
                m[i, j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            else:
                m[i, j] = levenshtein(smiles[i], smiles[j])

    # Complete the lower triangle
    m = m + m.T - np.diag(np.diag(m))
    m = 1 - m  # Convert distance to similarity
    np.fill_diagonal(m, 0)  # Set diagonal to 0

    return m


def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
    """Calculates a Tanimoto similarity matrix using ECFP fingerprints."""

    db_fp = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])

    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]], db_fp[smiles[j]])

    m = m + m.T - np.diag(np.diag(m))
    np.fill_diagonal(m, 0)

    return m


def get_scaffold_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
    """Calculates a Tanimoto similarity matrix based on generic Murcko scaffolds."""

    db_scaf = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(m)
        except Exception:
            print(f"Could not create a generic scaffold of {smi}, used a normal scaffold instead")
            skeleton = GetScaffoldForMol(m)
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        db_scaf[smi] = skeleton_fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])

    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]], db_scaf[smiles[j]])

    m = m + m.T - np.diag(np.diag(m))
    np.fill_diagonal(m, 0)

    return m
    

def detect_activity_cliffs_pairs(json_path: str):
    """Detect activity cliff pairs from a dataset using JSON config."""
    
    # Load parameters from JSON
    with open(json_path, "r") as f:
        params = json.load(f)
    
    input_path = params.get("input_path", "../../dataset/LNPs-TE/in vitro/AGILE HeLa.xlsx")
    output_train = params.get("output_train", "../../dataset/Activity Cliff/AGILE HeLa/train.xlsx")
    output_test = params.get("output_test", "../../dataset/Activity Cliff/AGILE HeLa/test.xlsx")
    output_pairs = params.get("output_pairs", "../../dataset/Activity Cliff/AGILE HeLa/activity_cliff_pairs.txt")
    
    similarity_threshold = params.get("similarity_threshold", 0.9)
    activity_diff_threshold = params.get("activity_diff_threshold", 1000)
    
    seed = params.get("seed", 42)
    set_seed(seed)

    # Load data
    df = pd.read_excel(input_path)
    df = df.dropna(subset=["Smiles", "Label"]).reset_index(drop=True)
    df['Original_Index'] = df.index

    smiles, labels = df["Smiles"].tolist(), df["Label"].tolist()

    # Precompute similarity matrices
    lev_matrix = get_levenshtein_matrix(smiles, normalize=True, hide=True)
    ecfp_matrix = get_tanimoto_matrix(smiles, radius=2, nBits=1024, hide=True)
    scaffold_matrix = get_scaffold_matrix(smiles, radius=2, nBits=1024, hide=True)

    # Composite similarity function
    def compute_similarity(i, j):
        return (lev_matrix[i][j] + ecfp_matrix[i][j] + scaffold_matrix[i][j]) / 3
    
    
    # Detect activity cliff pairs
    num_pairs = len(smiles) * (len(smiles) - 1) // 2
    cliff_pairs = {}

    for (i, j) in tqdm(combinations(range(len(smiles)), 2), total=num_pairs, desc="Detecting activity cliffs"):
        sim = compute_similarity(i, j)
        if sim < similarity_threshold:
            continue
        activity_i = 2 ** labels[i]
        activity_j = 2 ** labels[j]
        diff = abs(activity_i - activity_j)
        if diff > activity_diff_threshold:
            if i not in cliff_pairs or diff > cliff_pairs[i][2]:
                cliff_pairs[i] = (i, j, diff)
            if j not in cliff_pairs or diff > cliff_pairs[j][2]:
                cliff_pairs[j] = (j, i, diff)

    # Data splitting
    used_test_indices = set()
    pair_list = []
    for i, j, _ in cliff_pairs.values():
        if i in used_test_indices or j in used_test_indices:
            continue
        test = random.choice([i, j])
        train = j if test == i else i
        used_test_indices.add(test)
        pair_list.append((df.loc[train, 'Original_Index'], df.loc[test, 'Original_Index']))

    print(f"Final unique test pairs: {len(pair_list)}.")

    # Save pair information
    with open(output_pairs, "w") as f:
        f.write("train_index\ttest_index\ttrain_smiles\ttest_smiles\ttrain_label\ttest_label\t"
                "overall_similarity\tsmiles_similarity\tsubstructure_similarity\tscaffold_similarity\n")
        for train_idx, test_idx in pair_list:
            train_row = df[df['Original_Index'] == train_idx].iloc[0]
            test_row = df[df['Original_Index'] == test_idx].iloc[0]
            i, j = df.index.get_loc(train_idx), df.index.get_loc(test_idx)
            sim_lev = lev_matrix[i, j]
            sim_ecfp = ecfp_matrix[i, j]
            sim_scaf = scaffold_matrix[i, j]
            overall_similarity = (sim_lev + sim_ecfp + sim_scaf) / 3
            f.write(
                f"{train_idx}\t{test_idx}\t"
                f"{train_row['Smiles']}\t{test_row['Smiles']}\t"
                f"{train_row['Label']:.3f}\t{test_row['Label']:.3f}\t"
                f"{overall_similarity:.3f}\t{sim_lev:.3f}\t{sim_ecfp:.3f}\t{sim_scaf:.3f}\n"
            )

    # Construct training and test datasets
    test_raw_indices = [test for _, test in pair_list]
    test_df = df[df['Original_Index'].isin(test_raw_indices)].copy().set_index('Original_Index')
    train_df = df[~df['Original_Index'].isin(test_raw_indices)].copy().set_index('Original_Index')

    train_df.to_excel(output_train, index=True)
    test_df.to_excel(output_test, index=True)

    print(f"Training: {len(train_df)}, Test (from cliffs): {len(test_df)}, Pairs in txt file: {len(pair_list)}.")
    print("All the result files have been saved!")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Activity Cliff Data Splitter")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    args = parser.parse_args()
    
    detect_activity_cliffs_pairs(args.config)