import os
os.environ["PYTHONWARNINGS"] = "ignore"

import ast
import torch
import random
import numpy as np
import pandas as pd
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

from torch_geometric.loader import DataLoader 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset, TensorDataset

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


def load_data(df, task_type, chemberta, tokenizer, molclr, device, batch_size=256):
    """Convert SMILES to ECFP, ChemBERTa, MolCLR features and extra features tensor."""
    smiles = df["Smiles"].tolist()
    targets = df["Label"].values.astype(np.float32 if task_type=="regression" else np.int64)

    extra_features = df[[
        "Helper_Lipid_Index", "Gene_Index", "Cargo_Index",
        "Target_Index", "Type_Index", "Lipid_to_mRNA_Weight_Ratio",
        "Cationic_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio",
        "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio"
    ]].values.astype(np.float32)

    ecfp_features = smiles_to_ecfp(smiles)
    chemberta_features = smiles_to_chemberta(smiles, chemberta, tokenizer, device, batch_size)
    molclr_features = smiles_to_molclr(smiles, molclr, device, batch_size)

    X_ecfp = torch.tensor(ecfp_features, dtype=torch.float32).to(device)
    X_chemberta = torch.tensor(chemberta_features, dtype=torch.float32).to(device)
    X_molclr = torch.tensor(molclr_features, dtype=torch.float32).to(device)
    X_extra = torch.tensor(extra_features, dtype=torch.float32).to(device)
    y = torch.tensor(targets, dtype=torch.float32 if task_type=="regression" else torch.long).to(device)

    return X_ecfp, X_chemberta, X_molclr, X_extra, y
    

def train_fold(model, train_loader, criterion, optimizer, num_epochs, patiences, model_path):
    """Train one fold with early stopping."""
    best_loss = float("inf")
    count = 0

    for epoch in tqdm(range(num_epochs), desc="Training", leave=False):
        model.train()
        total_loss = 0
        for ecfp_f, chemberta_f, molclr_f, extra_f, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(ecfp_f, chemberta_f, molclr_f, extra_f)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            count = 0
        else:
            count += 1
            if count >= patiences:
                print(f"Early stopping at epoch {epoch + 1} with training loss: {avg_loss:.4f}.")
                break
    
    return model_path


def evaluate_fold(model, test_loader, task_type):
    """Evaluate one fold and return metrics dict."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, labels in test_loader:
            outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]
            if task_type=="classification":
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    if task_type=="classification":
        return {
            "Accuracy": accuracy_score(all_labels, all_preds),
            "Precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "Recall": recall_score(all_labels, all_preds, average="weighted"),
            "F1 Score": f1_score(all_labels, all_preds, average="weighted"),
            "ROC-AUC": roc_auc_score(all_labels, all_probs, multi_class="ovr")
        }
    else:
        return {
            "MSE": mean_squared_error(all_labels, all_preds),
            "MAE": mean_absolute_error(all_labels, all_preds),
            "R^2": r2_score(all_labels, all_preds),
            "PCC": pearsonr(all_labels, all_preds)[0]
        }


def cross_validation_run(task_type, dataset_names, 
    batch_size=128, learning_rate=1e-4, weight_decay=1e-2, 
    num_epochs=200, patiences=10, seed=42
    ):
    
    """Run scaffold-based cross-validation for regression or classification."""
    set_seed(seed)

    # Load pretrained models
    chemberta_model = "../../weights/ChemBERTa-77M-MTR/finetuned/"
    tokenizer = AutoTokenizer.from_pretrained(chemberta_model, clean_up_tokenization_spaces=True)
    chemberta = AutoModel.from_pretrained(chemberta_model).to(device)

    molclr_model = "../../weights/MolCLR/finetuned_gin/checkpoints/model.pth"
    molclr = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
    molclr.load_state_dict(torch.load(molclr_model, map_location=device))
    molclr = molclr.to(device)

    for dataset in dataset_names:
        print(f"\n{'=' * 50}")
        print(f"Cross Validation with {dataset} Sub-dataset")
        print(f"{'=' * 50}")

        df = pd.read_excel(f"../../dataset/LNPs-TE/task/{dataset}.xlsx")
        X_ecfp, X_chemberta, X_molclr, X_extra, y = load_data(df, task_type, chemberta, tokenizer, molclr, device)

        dataset_obj = LNPDataset(X_ecfp, X_chemberta, X_molclr, X_extra, y)
        splitter = Splitter(df)
        splits = splitter.get_splits(method="scaffold")

        results = (
            {"MSE": [], "MAE": [], "R^2": [], "PCC": []}
            if task_type=="regression" else
            {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "ROC-AUC": []}
        )

        for fold, indices in enumerate(splits):
            print(f"\n------ Processing Fold {fold + 1} ------")
            set_seed(seed)
            
            # Split dataset into training and validation sets
            train_idx = [idx for i in range(len(splits)) if i != fold for idx in splits[i]]
            test_idx = indices
                  
            print(f"Train size: {len(train_idx)}, test size: {len(test_idx)}")

            train_dataset = Subset(dataset_obj, train_idx)
            test_dataset = Subset(dataset_obj, test_idx)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(task_type=="classification"))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = LIFT(
                ecfp_dim=X_ecfp.shape[1], chemberta_dim=X_chemberta.shape[1],
                molclr_dim=X_molclr.shape[1], extra_dim=X_extra.shape[1],
                task_type=task_type, num_classes=len(np.unique(y.cpu())) if task_type=="classification" else 1
            ).to(device)

            criterion = nn.CrossEntropyLoss() if task_type=="classification" else nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay)

            model_path = f"../../ckpts/temp/Cross Validation/{dataset}/model_fold{fold+1}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            train_fold(model, train_loader, criterion, optimizer, num_epochs, patiences, model_path)
            model.load_state_dict(torch.load(model_path))

            fold_metrics = evaluate_fold(model, test_loader, task_type)
            for k, v in fold_metrics.items():
                results[k].append(v)

            fold_metrics = {k: f"{v:.3f}" for k, v in fold_metrics.items()}
            print(f"Fold {fold+1}: {fold_metrics}")

        print("\n------ Metrics Summary ------")
        for k, vals in results.items():
            print(f"{k} - Mean: {np.mean(vals):.3f}, Std: {np.std(vals):.3f}")
   

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run Cross-Validation Experiments.")

    # Optional JSON config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    # Command-line arguments
    parser.add_argument("--task", type=str, choices=["classification", "regression"], help="Task type")
    parser.add_argument("--datasets", type=str, nargs="+", help="Dataset names")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--patiences", type=int, help="Early stopping patience")
    parser.add_argument("--gpu", type=str, help="GPU device, e.g., '0', 'cuda:0', or 'cpu'")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Load JSON config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    # Final parameters selection
    task_type = args.task or config.get("task", "regression")
    dataset_names = args.datasets or config.get("datasets", [])
    batch_size = args.batch_size or config.get("batch_size", 128)
    learning_rate = args.learning_rate or config.get("learning_rate", 1e-4)
    weight_decay = args.weight_decay or config.get("weight_decay", 1e-2)
    num_epochs = args.num_epochs or config.get("num_epochs", 200)
    patiences = args.patiences or config.get("patiences", 10)
    gpu_idx = args.gpu or config.get("gpu", None)
    seed = args.seed or config.get("seed", 42)
    
    if gpu_idx is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_idx}")   
    print(f"Using device: {device}")

    # Cross Validation
    cross_validation_run(
        task_type=task_type,
        dataset_names=dataset_names,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patiences=patiences,
        seed=seed
    )