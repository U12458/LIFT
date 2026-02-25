import os
os.environ["PYTHONWARNINGS"] = "ignore"

import ast
import torch
import random
import logging
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import torch.nn as nn
from rdkit import Chem
import torch.optim as optim
import torch.nn.functional as F

from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

from joblib import Parallel, delayed
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from torch_geometric.loader import DataLoader 
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
    )


from utils.featurizers import *
from models.dataset import LNPDataset
from models.feature_fusion import LIFT


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

transformers.logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 42 
set_seed(seed)


# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


chemberta_model = "./weights/ChemBERTa-77M-MTR/finetuned/"
tokenizer = AutoTokenizer.from_pretrained(chemberta_model, clean_up_tokenization_spaces=True)
chemberta = AutoModel.from_pretrained(chemberta_model).to(device)


molclr_model = "./weights/MolCLR/finetuned_gin/checkpoints/model.pth"
molclr = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
molclr.load_state_dict(torch.load(molclr_model, map_location=device))
molclr = molclr.to(device)


def load_dataset(dataset_name, task_type='regression', device='cuda', seed=42):
    """
    Load dataset from Excel file, generate embeddings,
    split into train/validation/test sets,
    and convert data into PyTorch tensors.
    """
    data_file = f"./dataset/LNPs-TE/task/{dataset_name}.xlsx"
    df = pd.read_excel(data_file)
    
    # Extract SMILES strings and target labels
    smiles = df["Smiles"].tolist()
    targets = df["Label"].values
    if task_type == 'regression':
        targets = targets.astype(np.float32)
    else:
        targets = targets.astype(np.int64)
    
    # Extract extra features
    extra_features = df[[
        "Helper_Lipid_Index", "Gene_Index", "Cargo_Index",
        "Target_Index", "Type_Index", "Lipid_to_mRNA_Weight_Ratio", 
        "Cationic_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio", 
        "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio"
    ]].values.astype(np.float32)
    
    # Generate embeddings for different representations
    ecfp_features = smiles_to_ecfp(smiles)
    chemberta_features = smiles_to_chemberta(smiles, chemberta, tokenizer, device, batch_size=256)
    molclr_features = smiles_to_molclr(smiles, molclr, device, batch_size=256)

    # Split data into training, validation and test
    X_train_ecfp, X_temp_ecfp, X_train_chemberta, X_temp_chemberta, \
    X_train_molclr, X_temp_molclr, X_train_extra, X_temp_extra, \
    y_train, y_temp = train_test_split(
        ecfp_features, chemberta_features, molclr_features, extra_features, targets, 
        test_size=0.2, random_state=seed
    )
    X_valid_ecfp, X_test_ecfp, X_valid_chemberta, X_test_chemberta, \
    X_valid_molclr, X_test_molclr, X_valid_extra, X_test_extra, \
    y_valid, y_test = train_test_split(
        X_temp_ecfp, X_temp_chemberta, X_temp_molclr, X_temp_extra, y_temp, 
        test_size=0.5, random_state=seed
    )

    # Convert arrays to PyTorch tensors
    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype).to(device)
    
    X_train_ecfp, X_valid_ecfp, X_test_ecfp = map(to_tensor, [X_train_ecfp, X_valid_ecfp, X_test_ecfp])
    X_train_chemberta, X_valid_chemberta, X_test_chemberta = map(to_tensor, [X_train_chemberta, X_valid_chemberta, X_test_chemberta])
    X_train_molclr, X_valid_molclr, X_test_molclr = map(to_tensor, [X_train_molclr, X_valid_molclr, X_test_molclr])
    X_train_extra, X_valid_extra, X_test_extra = map(to_tensor, [X_train_extra, X_valid_extra, X_test_extra])
    
    # Convert target labels
    if task_type == 'regression':
        y_train, y_valid, y_test = map(lambda x: to_tensor(x, dtype=torch.float32), [y_train, y_valid, y_test])
    else:
        y_train, y_valid, y_test = map(lambda x: to_tensor(x, dtype=torch.long), [y_train, y_valid, y_test])
    
    # Dimensional information
    dims = {
        'ecfp_dim': ecfp_features.shape[1],
        'chemberta_dim': chemberta_features.shape[1],
        'molclr_dim': molclr_features.shape[1],
        'extra_dim': extra_features.shape[1],
        'num_classes': len(np.unique(targets)) if task_type=='classification' else 1
    }

    train = (X_train_ecfp, X_train_chemberta, X_train_molclr, X_train_extra, y_train)
    valid = (X_valid_ecfp, X_valid_chemberta, X_valid_molclr, X_valid_extra, y_valid)
    test  = (X_test_ecfp, X_test_chemberta, X_test_molclr, X_test_extra, y_test)

    return train, valid, test, dims


def evaluation(model_class, model_path, test_data, task_type='regression', device='cuda'):

    X_test_ecfp, X_test_chemberta, X_test_molclr, X_test_extra, y_test = test_data
    test_dataset = LNPDataset(X_test_ecfp, X_test_chemberta, X_test_molclr, X_test_extra, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = model_class.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels, all_probs, all_attn = [], [], [], []

    with torch.no_grad():
        for ecfp, chemberta, molclr, extra, labels in test_loader:
            outputs, attn = model(ecfp, chemberta, molclr, extra)
            all_attn.extend(attn.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if task_type == 'regression':
                all_preds.extend(outputs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    all_attn = np.array(all_attn)
    attns = np.mean(all_attn, axis=0)

    metrics = {}
    if task_type == 'regression':
        metrics['MSE'] = mean_squared_error(all_labels, all_preds)
        metrics['MAE'] = mean_absolute_error(all_labels, all_preds)
        metrics['R^2'] = r2_score(all_labels, all_preds)
        metrics['PCC'] = pearsonr(all_labels, all_preds)[0]
    else:
        all_probs = np.array(all_probs)
        metrics['Accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['Precision'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(all_labels, all_preds, average='weighted')
        metrics['F1 Score'] = f1_score(all_labels, all_preds, average='weighted')

        num_classes = all_probs.shape[1]
        y_bin = label_binarize(all_labels, classes=np.arange(num_classes))
        metrics['ROC-AUC'] = roc_auc_score(y_bin, all_probs, multi_class="ovr")

    return metrics, attns


datasets = {
    'regression': ["RedAm", "BCar", "AGILE HeLa", "AGILE RAW", "BEst A549", "Wh HeLa"],
    'classification': [
        "Akinc", "iPhos", "3CR HeLa", "Den HeLa",
        "Thio", "3CR Est", "4CR HeLa", "BEst Liver"
    ]
}

num_repeats = 3


for task_type, dataset_list in datasets.items():
    print(f"\n{'#' * 30}")
    print(f"====== {task_type.capitalize()} Task ======")
    print(f"{'#' * 30}")
    
    for dataset in dataset_list:
        train, valid, test, dims = load_dataset(dataset, task_type=task_type, device=device, seed=seed)

        print(f"\n{'=' * 25}")
        print(f"Sub-dataset: {dataset}")
        print(f"{'=' * 25}")
        
        all_metrics, all_attns = [], []
        
        for repeat in range(num_repeats):
        
            ckpt_path = f"./ckpts/{task_type.capitalize()}/{dataset}/model_repeat{repeat+1}.pth"
            model = LIFT(
                ecfp_dim=dims['ecfp_dim'], chemberta_dim=dims['chemberta_dim'],
                molclr_dim=dims['molclr_dim'], extra_dim=dims['extra_dim'],
                task_type=task_type, num_classes=dims['num_classes']
            )
            
            metrics, attns = evaluation(
                model, ckpt_path, 
                test, task_type=task_type, 
                device=device
            )
            
            all_metrics.append(metrics)
            all_attns.append(attns)

            formatted_metrics = {k: f"{v:.3f}" for k, v in metrics.items()}
            print(f"Repeat {repeat+1} - Metrics: {formatted_metrics}")
        
        # Compute mean and std for metrics
        metric_keys = all_metrics[0].keys()
        print(f"\n--- Repeat Experimental Results ---")
        for k in metric_keys:
            values = [m[k] for m in all_metrics if isinstance(m[k], (int, float))]
            mean, std = np.mean(values), np.std(values)
            print(f"{k} - Mean: {mean:.3f}, Std: {std:.3f}")
        
        # Compute mean and std for feature attention weights
        attn_array = np.array(all_attns)
        attn_means = attn_array.mean(axis=0)
        attn_stds = attn_array.std(axis=0)
        
        print(f"\n--- Weighting Factor Summary ---")
        print(f"ECFP(Radius=9)    - Mean: {attn_means[0]:.3f}, Std: {attn_stds[0]:.3f}")
        print(f"ChemBERTa-2(MTR)* - Mean: {attn_means[1]:.3f}, Std: {attn_stds[1]:.3f}")
        print(f"MolCLR(GIN)*      - Mean: {attn_means[2]:.3f}, Std: {attn_stds[2]:.3f}")