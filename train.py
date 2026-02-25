import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel

from utils.featurizers import *
from models.dataset import LNPDataset
from models.feature_fusion import LIFT

import logging, warnings, transformers
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

seed = 42 
set_seed(seed)


# Dataset Loading
def load_dataset(dataset_path, task_type, device, chemberta, tokenizer, molclr, test_size=0.2, val_size=0.5):
    """Load dataset, compute features, and split into train/val/test."""
    df = pd.read_excel(dataset_path)
    smiles = df["Smiles"].tolist()
    targets = df["Label"].values.astype(np.float32 if task_type=="regression" else np.int64)
    extra_features = df[[
        "Helper_Lipid_Index", "Gene_Index", "Cargo_Index",
        "Target_Index", "Type_Index", "Lipid_to_mRNA_Weight_Ratio",
        "Cationic_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio",
        "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio"
    ]].values.astype(np.float32)
    
    num_classes = len(np.unique(targets)) if task_type == "classification" else 1

    # Generate features
    ecfp_features = smiles_to_ecfp(smiles, radius=9)
    chemberta_features = smiles_to_chemberta(smiles, chemberta, tokenizer, device, batch_size=256)
    molclr_features = smiles_to_molclr(smiles, molclr, device, batch_size=256)

    ecfp_dim, chemberta_dim, molclr_dim, extra_dim = (
        ecfp_features.shape[1], chemberta_features.shape[1],
        molclr_features.shape[1], extra_features.shape[1]
    )
    
    feature_dims = {
        "ecfp_dim": ecfp_dim,
        "chemberta_dim": chemberta_dim,
        "molclr_dim": molclr_dim,
        "extra_dim": extra_dim
    }

    # Split sub-dataset with training:validation:test = 8:1:1
    X_train_ecfp, X_temp_ecfp, X_train_chemberta, X_temp_chemberta, \
    X_train_molclr, X_temp_molclr, X_train_extra, X_temp_extra, \
    y_train, y_temp = train_test_split(
        ecfp_features, chemberta_features, molclr_features,
        extra_features, targets, test_size=test_size, random_state=seed
    )

    X_valid_ecfp, X_test_ecfp, X_valid_chemberta, X_test_chemberta, \
    X_valid_molclr, X_test_molclr, X_valid_extra, X_test_extra, \
    y_valid, y_test = train_test_split(
        X_temp_ecfp, X_temp_chemberta, X_temp_molclr,
        X_temp_extra, y_temp, test_size=val_size, random_state=seed
    )

    def to_tensor(arr, is_label=False):
        dtype = torch.long if (task_type=="classification" and is_label) else torch.float32
        return torch.tensor(arr, dtype=dtype).to(device)

    X_train_ecfp, X_valid_ecfp, X_test_ecfp = map(to_tensor, [X_train_ecfp, X_valid_ecfp, X_test_ecfp])
    X_train_chemberta, X_valid_chemberta, X_test_chemberta = map(to_tensor, [X_train_chemberta, X_valid_chemberta, X_test_chemberta])
    X_train_molclr, X_valid_molclr, X_test_molclr = map(to_tensor, [X_train_molclr, X_valid_molclr, X_test_molclr])
    X_train_extra, X_valid_extra, X_test_extra = map(to_tensor, [X_train_extra, X_valid_extra, X_test_extra])
    y_train, y_valid, y_test = map(lambda x: to_tensor(x, is_label=True), [y_train, y_valid, y_test])

    train_dataset = LNPDataset(X_train_ecfp, X_train_chemberta, X_train_molclr, X_train_extra, y_train)
    valid_dataset = LNPDataset(X_valid_ecfp, X_valid_chemberta, X_valid_molclr, X_valid_extra, y_valid)
    test_dataset = LNPDataset(X_test_ecfp, X_test_chemberta, X_test_molclr, X_test_extra, y_test)

    return train_dataset, valid_dataset, test_dataset, feature_dims, num_classes


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, model_path):
    """Train model and save best checkpoint based on validation loss."""
    best_valid_loss = float("inf")
    for epoch in tqdm(range(num_epochs), desc="Training", leave=False):
        model.train()
        for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, labels in valid_loader:
                outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]
                valid_loss += criterion(outputs, labels).item()
        valid_loss /= len(valid_loader)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
    
    return model


def test_model(model, test_loader, task_type):
    """Evaluate model and return metrics dictionary."""
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


def run(task_type, dataset_names, batch_size=32, learning_rate=1e-4, weight_decay=1e-4,
        num_epochs=200, num_repeats=3, seed=42):
    """Main experiment runner for classification or regression."""
    set_seed(seed)

    # Load pre-trained models
    chemberta_model = "./weights/ChemBERTa-77M-MTR/finetuned/"
    tokenizer = AutoTokenizer.from_pretrained(chemberta_model, clean_up_tokenization_spaces=True)
    chemberta = AutoModel.from_pretrained(chemberta_model).to(device)

    molclr_model = "./weights/MolCLR/finetuned_gin/checkpoints/model.pth"
    molclr = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
    molclr.load_state_dict(torch.load(molclr_model, map_location=device))
    molclr = molclr.to(device)

    for dataset in dataset_names:
        print(f"\n{'=' * 40}")
        print(f"{task_type.capitalize()} Task on {dataset} Sub-dataset")
        print(f"{'=' * 40}")
        
        train_dataset, valid_dataset, test_dataset, dims, num_classes = load_dataset(
            f"dataset/LNPs-TE/task/{dataset}.xlsx", 
            task_type, device, chemberta, tokenizer, molclr
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = {key: [] for key in (
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"] if task_type=="classification" 
            else ["MSE", "MAE", "R^2", "PCC"]
        )}

        for repeat in range(num_repeats):

            model = LIFT(
                ecfp_dim=dims["ecfp_dim"],
                chemberta_dim=dims["chemberta_dim"],
                molclr_dim=dims["molclr_dim"],
                extra_dim=dims["extra_dim"],
                task_type=task_type,
                num_classes=num_classes
            ).to(device)

            criterion = nn.CrossEntropyLoss() if task_type=="classification" else nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay)
            
            model_path = f"./ckpts/temp/{dataset}/model_{repeat+1}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, model_path)
            model.load_state_dict(torch.load(model_path))
            metrics = test_model(model, test_loader, task_type)

            for k, v in metrics.items():
                results[k].append(v)
            
            print(f"Repeat {repeat+1} - {', '.join([f'{k}: {v:.3f}' for k,v in metrics.items()])}")

        # Summary for dataset
        print("\n*** Metrics Summary ***")
        for k, vals in results.items():
            mean, std = np.mean(vals), np.std(vals)
            print(f"{k} - Mean: {mean:.3f}, Std: {std:.3f}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LIFT experiments for Classification or Regression tasks.")
    
    # Optional JSON configuration file
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    # Command-line arguments can override JSON configuration
    parser.add_argument("--task", type=str, choices=["classification", "regression"], help="Task type")
    parser.add_argument("--datasets", type=str, nargs="+", help="Dataset names")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--num_repeats", type=int, help="Number of repeated experiments")
    parser.add_argument("--gpu", type=str, help="GPU device, e.g., '0', 'cuda:0', or 'cpu'")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()

    # Load JSON config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    # Final parameter selection
    task_type = args.task or config.get("task", "regression")
    dataset_names = args.datasets or config.get("datasets", [])
    batch_size = args.batch_size or config.get("batch_size", 32)
    learning_rate = args.learning_rate or config.get("learning_rate", 1e-4)
    weight_decay = args.weight_decay or config.get("weight_decay", 1e-4)
    num_epochs = args.num_epochs or config.get("num_epochs", 200)
    num_repeats = args.num_repeats or config.get("num_repeats", 3)
    gpu_idx = args.gpu or config.get("gpu", None)
    seed = args.seed or config.get("seed", 42)
    
    if gpu_idx is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_idx}")   
    print(f"Using device: {device}")

    # Run the experiment
    run(
        task_type=task_type,
        dataset_names=dataset_names,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        num_repeats=num_repeats,
        seed=seed
    )