import random
import numpy as np
from rdkit import Chem
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold


class Splitter:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def generate_scaffold(self, smiles, include_chirality=False):

        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

        return scaffold

    def scaffold_split(self):
        
        scaffold_to_indices = defaultdict(list)
    
        # Generate scaffold for each molecule and group indices by scaffold
        for idx, smiles in enumerate(self.dataframe["Smiles"]):
            scaffold = self.generate_scaffold(smiles)
            scaffold_to_indices[scaffold].append(idx)

        total_samples = len(self.dataframe)
        min_size = int(total_samples * 0.1)  # Minimum fold size: 10% of the total dataset

        large_folds = []  # Scaffolds with >= 10% of the total samples
        small_scaffold_indices = []  # Scaffolds with < 10% of the total samples

        for scaffold, indices in scaffold_to_indices.items():
            if len(indices) >= min_size:
                large_folds.append(indices)
            else:
                small_scaffold_indices.append(indices)

        small_folds = []
        current_fold = []
        current_count = 0

        # Sort by scaffold size with descending
        small_scaffold_indices.sort(key=lambda x: len(x), reverse=True)

        for indices in small_scaffold_indices:
            current_fold.extend(indices)
            current_count += len(indices)
            if current_count >= min_size:
                small_folds.append(current_fold)
                current_fold = []
                current_count = 0

        folds = large_folds + small_folds

        print(f"Split the dataset with Scaffold method for {len(folds)}-Fold Cross Validation.")
        
        return folds

    def random_split(self, seed=42):

        n_splits = 5
        print(f"Split the dataset with Random method for {n_splits}-Fold Cross Validation.")

        random.seed(seed)
        indices = list(range(len(self.dataframe)))
        random.shuffle(indices)
        
        return np.array_split(indices, n_splits)

    def get_splits(self, method='scaffold'):
        if method == 'scaffold':
            return self.scaffold_split()
        elif method == 'random':
            return self.random_split()
        else:
            raise ValueError("Invalid splitting method. Choose 'scaffold' or 'random'.") 