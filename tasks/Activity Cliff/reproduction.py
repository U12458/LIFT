import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr
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


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(file_path: str):
    """Load dataset from Excel file and extract features and labels."""
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
    """Extract multidimensional molecular features."""
    ecfp_features = smiles_to_ecfp(smiles_list)
    chemberta_features = smiles_to_chemberta(smiles_list, chemberta, tokenizer, device, batch_size)
    molclr_features = smiles_to_molclr(smiles_list, molclr, device, batch_size)
    
    return ecfp_features, chemberta_features, molclr_features


def inference(model, dataset, smiles_list, file_path, batch_size=32, device=None):
    """Perform inference and save predicted results."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = [], []

    with torch.no_grad():
        for ecfp_feature, chemberta_feature, molclr_feature, extra_feature, label in tqdm(loader, desc="Inference", leave=False):
            outputs = model(ecfp_feature, chemberta_feature, molclr_feature, extra_feature)[0]
            preds.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy())
            
    # Flatten arrays before metric computation
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    # Compute evaluation metrics
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson_corr = pearsonr(labels, preds)[0]

    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f}, PCC: {pearson_corr:.3f}")

    # Save predictions to Excel
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(dir_name, f"{base_name}_predictions.xlsx")

    df = pd.DataFrame({
        "SMILES": smiles_list,
        "Label": labels,
        "Prediction": [p[0] if isinstance(p, (list, np.ndarray)) else p for p in preds]
    })
    os.makedirs(dir_name, exist_ok=True)
    df.to_excel(save_path, index=False)
    
    print(f"Predictions have been saved to: {save_path}")


if __name__ == "__main__":
    
    # Configuration
    datasets = ["AGILE HeLa", "AGILE RAW"]
    base_data_path = "../../dataset/Activity Cliff"
    base_ckpt_path = "../../ckpts/Activity Cliff"
    
    # Initialization
    seed = 42
    set_seed(seed)
    batch_size=128
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load pre-trained models
    chemberta_model = "../../weights/ChemBERTa-77M-MTR/finetuned/"
    tokenizer = AutoTokenizer.from_pretrained(chemberta_model, clean_up_tokenization_spaces=True)
    chemberta = AutoModel.from_pretrained(chemberta_model).to(device)

    molclr_model = "../../weights/MolCLR/finetuned_gin/checkpoints/model.pth"
    molclr = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
    molclr.load_state_dict(torch.load(molclr_model, map_location=device))
    molclr = molclr.to(device)

    # Loop through each dataset
    for dataset in datasets:
        
        print(f"\n" + "="*50)
        print(f"Processing Dataset: {dataset}")
        print("="*50)

        # Construct dynamic paths
        train_path = f"{base_data_path}/{dataset}/train.xlsx"
        test_path = f"{base_data_path}/{dataset}/test.xlsx"
        ckpt_path = f"{base_ckpt_path}/{dataset}.pth"

        # Load data
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

        # Initialize LIFT Model
        model = LIFT(
            ecfp_dim=train_ecfp.shape[1],
            chemberta_dim=train_chemberta.shape[1],
            molclr_dim=train_molclr.shape[1],
            extra_dim=train_extra.shape[1],
            task_type="regression",
            num_classes=1
        ).to(device)

        # Load checkpoint
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, skipping...")
            continue

        # Run inference
        print(f"\n===== Inference on Training Set =====")
        inference(model, train_dataset, train_smiles, train_path, batch_size, device)

        print(f"\n===== Inference on Test Set =====")
        inference(model, test_dataset, test_smiles, test_path, batch_size, device)