import os
os.environ["PYTHONWARNINGS"] = "ignore"

import ast
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

from joblib import Parallel, delayed
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from torch_geometric.loader import DataLoader 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
    )

from utils.featurizers import *
from utils.data_splitter import Splitter
from models.dataset import LNPDataset
from models.feature_fusion import LIFT


import logging
import transformers

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


def preprocess_dataset(file_path, task_type, chemberta, tokenizer, molclr, device):
    """Load dataset, compute features and create LNPDataset."""
    df = pd.read_excel(file_path)
    smiles = df["Smiles"].tolist()
    targets = df["Label"].values.astype(np.float32 if task_type == "regression" else np.int64)
    
    extra_features = df[[
        "Helper_Lipid_Index", "Gene_Index", "Cargo_Index",
        "Target_Index", "Type_Index", "Lipid_to_mRNA_Weight_Ratio",
        "Cationic_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio",
        "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio"
    ]].values.astype(np.float32)

    # Compute features
    ecfp_features = smiles_to_ecfp(smiles)
    chemberta_features = smiles_to_chemberta(smiles, chemberta, tokenizer, device, batch_size=256)
    molclr_features = smiles_to_molclr(smiles, molclr, device, batch_size=256)

    dataset = LNPDataset(
        torch.tensor(ecfp_features, dtype=torch.float32).to(device),
        torch.tensor(chemberta_features, dtype=torch.float32).to(device),
        torch.tensor(molclr_features, dtype=torch.float32).to(device),
        torch.tensor(extra_features, dtype=torch.float32).to(device),
        torch.tensor(targets, dtype=torch.float32 if task_type=="regression" else torch.long).to(device)
    )

    dims = (ecfp_features.shape[1], chemberta_features.shape[1],
            molclr_features.shape[1], extra_features.shape[1])
    
    return df, dataset, dims


def evaluation(model, loader, task_type="regression", device="cuda"):
    """
    Evaluate model on a dataset loader for tasks.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, labels in loader:
            outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]

            if task_type == "regression":
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            elif task_type == "classification":
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    if task_type == "regression":
        return {
            "MSE": mean_squared_error(all_labels, all_preds),
            "MAE": mean_absolute_error(all_labels, all_preds),
            "R^2": r2_score(all_labels, all_preds),
            "PCC": pearsonr(all_labels, all_preds)[0]
        }

    elif task_type == "classification":
        return {
            "Accuracy": accuracy_score(all_labels, all_preds),
            "Precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "Recall": recall_score(all_labels, all_preds, average="weighted"),
            "F1 Score": f1_score(all_labels, all_preds, average="weighted"),
            "ROC-AUC": roc_auc_score(all_labels, all_probs, multi_class="ovr")
        }


def cross_validation(df, dataset, dims, device, task_type, batch_size=32, model_ckpt_path=None):
    """Run scaffold-based cross-validation."""
    splitter = Splitter(df)
    splits = splitter.get_splits(method="scaffold")

    results = {}
    if task_type == "regression":
        metrics_keys = ["MSE", "MAE", "R^2", "PCC"]
    else:
        metrics_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    results = {k: [] for k in metrics_keys}

    for fold, indices in enumerate(splits):
        train_idx = [idx for i, split in enumerate(splits) if i != fold for idx in split]
        test_idx = indices
        
        print(f"Train size: {len(train_idx)}, test size: {len(test_idx)}")
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

        # Initialize model
        num_classes = len(np.unique(df["Label"])) if task_type=="classification" else 1
        model = LIFT(*dims, task_type=task_type, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(f"{model_ckpt_path}/model_fold{fold+1}.pth"))
        
        # Evaluate
        metrics = evaluation(model, test_loader, task_type=task_type)
        for k, v in metrics.items():
            results[k].append(v)
        
        print(f"Fold {fold + 1} - {', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}")


    # Compute mean and std
    summary = {k: (np.mean(v), np.std(v)) for k, v in results.items()}
    print("\n------ Cross-Validation Summary ------")
    for k, (mean_val, std_val) in summary.items():
        print(f"{k} - Mean: {mean_val:.3f}, Std: {std_val:.3f}")

    return results


if __name__ == "__main__":
    set_seed(42)

    # Load pretrained models
    chemberta_model = "../../weights/ChemBERTa-77M-MTR/finetuned/"
    tokenizer = AutoTokenizer.from_pretrained(chemberta_model, clean_up_tokenization_spaces=True)
    chemberta = AutoModel.from_pretrained(chemberta_model).to(device)

    molclr_model = "../../weights/MolCLR/finetuned_gin/checkpoints/model.pth"
    molclr = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
    molclr.load_state_dict(torch.load(molclr_model, map_location=device))
    molclr = molclr.to(device)

    # Define dataset
    Datasets = {
        "regression": ["AGILE HeLa", "AGILE RAW"],
        "classification": ["iPhos", "Den HeLa"]
    }

    # Process all datasets
    for task_type, datasets in Datasets.items():
        for index in datasets:
            print(f"\n{'='*25}")
            print(f"Sub-dataset: {index}")
            print(f"{'='*25}")
            file_path = f"../../dataset/LNPs-TE/task/{index}.xlsx"
            ckpt_path = f"../../ckpts/Cross Validation/{index}"

            df, dataset, dims = preprocess_dataset(
                file_path, task_type, 
                chemberta, tokenizer, molclr, device
            )

            cross_validation(
                df, dataset, dims, device, 
                task_type, batch_size=32,
                model_ckpt_path=ckpt_path
            )