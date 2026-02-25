import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import ast
import json
import glob
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem
import torch.optim as optim
import torch.nn.functional as F
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

from joblib import Parallel, delayed
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import Subset, Dataset, TensorDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.featurizers import *
from models.dataset import LNPDataset
from models.feature_fusion import LIFT

import logging
import transformers

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

transformers.logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

torch.use_deterministic_algorithms(True)


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(file_path: str):
    """Load and process dataset."""
    data = pd.read_excel(file_path)

    extra_columns = [
        "Helper_Lipid_Index", "Gene_Index", "Cargo_Index",
        "Target_Index", "Type_Index", "Lipid_to_mRNA_Weight_Ratio",
        "Cationic_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio",
        "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio"
    ]

    smiles = data["Smiles"].tolist()
    extras = data[extra_columns].to_numpy(dtype=np.float32)
    labels = data["Label"].values.astype(np.float32)

    return smiles, extras, labels
    

def extract_features(smiles_list, device, batch_size=256):
    """
    Extract multidimensional molecular features for a given SMILES list.
    """
    ecfp_features = smiles_to_ecfp(smiles_list)
    chemberta_features = smiles_to_chemberta(smiles_list, chemberta, tokenizer, device, batch_size=256)
    molclr_features = smiles_to_molclr(smiles_list, molclr, device, batch_size=256)
    
    return ecfp_features, chemberta_features, molclr_features


def train(model, train_loader, criterion, optimizer, num_epochs=200, patience=10, best_model_path=None):
    """
    Train the model with early stopping.
    """
    best_train_loss = float('inf')
    count = 0

    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=False):
        model.train()
        total_loss = 0.0

        for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
            count = 0
        else:
            count += 1
            if count >= patience:
                print(f"Early stopping at epoch {epoch + 1} with training loss: {train_loss:.4f}.")
                break

    print(f"Training finished. Best training loss: {best_train_loss:.4f}")
    
    return model


def evaluate(model, test_loader, device=None):
    """Evaluate the model and compute metrics."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, labels in test_loader:
            outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_corr = pearsonr(all_labels, all_preds)[0]

    print(f"------ Evaluation on Test Dataset ------")
    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f}, PCC: {pearson_corr:.3f}")
    
    return all_preds, all_labels
    

def activity_cliff(
    train_path, test_path,
    batch_size=32, learning_rate=1e-4, weight_decay=1e-4,
    num_epochs=200, patience=10,
    seed=42, device=None,
    best_model_path="../../ckpts/temp/AGILE HeLa/activity_cliff.pth"
):
    
    """
    Run LIFT training and evaluation.
    """

    # Load datasets
    train_smiles, train_extra, train_labels = load_dataset(train_path)
    test_smiles, test_extra, test_labels = load_dataset(test_path)
    
    # Feature extraction
    print("Extracting training dataset features...")
    train_ecfp, train_chemberta, train_molclr = extract_features(
        train_smiles, device, batch_size=batch_size
    )
    print("Training dataset features extraction completed.")
    
    print("Extracting test dataset features...")
    test_ecfp, test_chemberta, test_molclr = extract_features(
        test_smiles, device, batch_size=batch_size
    )
    print("Test dataset features extraction completed.")
    
    # Convert to tensors and create DataLoader
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(device)

    train_dataset = LNPDataset(
        to_tensor(train_ecfp), to_tensor(train_chemberta),
        to_tensor(train_molclr), to_tensor(train_extra),
        to_tensor(train_labels)
    )

    test_dataset = LNPDataset(
        to_tensor(test_ecfp), to_tensor(test_chemberta),
        to_tensor(test_molclr), to_tensor(test_extra),
        to_tensor(test_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LIFT(
        ecfp_dim=train_ecfp.shape[1],
        chemberta_dim=train_chemberta.shape[1],
        molclr_dim=train_molclr.shape[1],
        extra_dim=train_extra.shape[1],
        task_type="regression",
        num_classes=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate,
        betas=(0.9, 0.99), weight_decay=weight_decay
    )
    
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Train model
    model = train(
        model, train_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=patience,
        best_model_path=best_model_path
    )

    # Evaluate model
    model.load_state_dict(torch.load(best_model_path))
    
    evaluate(model, test_loader, device=device)
    

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    parser = argparse.ArgumentParser(description="Activity Cliff Task")
    
    # Optional JSON config file
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    
    # CLI arguments (override JSON)
    parser.add_argument("--train_path", type=str, nargs="+")
    parser.add_argument("--test_path", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weight_path", type=str)

    args = parser.parse_args()

    # Load JSON config (optional)
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    # Final parameter resolution
    train_path = args.train_path or config.get("train_path", [])
    test_path = args.test_path or config.get("test_path", [])

    batch_size = args.batch_size or config.get("batch_size", 32)
    learning_rate = args.learning_rate or config.get("learning_rate", 1e-4)
    weight_decay = args.weight_decay or config.get("weight_decay", 1e-4)
    
    num_epochs = args.num_epochs or config.get("num_epochs", 200)
    patience = args.patience or config.get("patience", 10)
    seed = args.seed or config.get("seed", 42)

    weight_path = args.weight_path or config.get(
        "weight_path", "../../ckpts/temp/AGILE HeLa/activity_cliff.pth"
    )

    # Device selection
    gpu_idx = args.gpu or config.get("gpu", None)
    if gpu_idx is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_idx}")

    print(f"Using device: {device}")

    # Load pre-trained models
    chemberta_model = "../../weights/ChemBERTa-77M-MTR/finetuned/"
    tokenizer = AutoTokenizer.from_pretrained(chemberta_model, clean_up_tokenization_spaces=True)
    chemberta = AutoModel.from_pretrained(chemberta_model).to(device)

    molclr_model = "../../weights/MolCLR/finetuned_gin/checkpoints/model.pth"
    molclr = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
    molclr.load_state_dict(torch.load(molclr_model, map_location=device))
    molclr = molclr.to(device)

    # Run main
    activity_cliff(
        train_path=train_path,
        test_path=test_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        seed=seed,
        device=device,
        best_model_path=weight_path
    )