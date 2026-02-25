import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel

from .utils import *
from .molclr_model import *


def smiles_to_ecfp(smiles, nBits=1024, radius=9):

    ecfp_list = []
    for smile in tqdm(smiles, desc="Processing ECFP features", leave=False):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            ecfp_list.append(np.zeros(nBits, dtype=np.float32))
        else:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            ecfp = generator.GetFingerprint(mol)
            ecfp_list.append(np.array(ecfp, dtype=np.float32))
            
    ecfp_features = np.array(ecfp_list, dtype=np.float32)
    
    return ecfp_features


def smiles_to_chemberta(smiles, chemberta, tokenizer, device, batch_size=256):
        
    chemberta.eval()
    embeddings = []

    with tqdm(total=len(smiles), desc="Processing ChemBERTa features", leave=False) as t:
        for i in range(0, len(smiles), batch_size):
            batch_smiles = smiles[i : i + batch_size]
            tokens = tokenizer(
                batch_smiles, padding=True, truncation=True, 
                return_tensors="pt", max_length=512
                )
            tokens = {key: val.to(device) for key, val in tokens.items()}

            with torch.no_grad():
                outputs = chemberta(**tokens)
                
            embeddings.append(outputs.pooler_output.cpu().numpy())
            t.update(len(batch_smiles))
    
    chemberta_features = np.vstack(embeddings)
    
    return chemberta_features
    

def smiles_to_molclr(smiles, molclr, device, batch_size=256):

    molclr.eval()
    embeddings = []

    with tqdm(total=len(smiles), desc="Processing MoLCLR features", leave=False) as t:
        for i in range(0, len(smiles), batch_size):
            
            batch_smiles = smiles[i : i + batch_size]
            batch_graphs = smiles_list_to_graph(batch_smiles, device)
            batch_loader = DataLoader(batch_graphs, batch_size=batch_size)

            batch_embeddings = []
            with torch.no_grad():
                for batch in batch_loader:
                    batch = batch.to(device)
                    h, _ = molclr(batch)
                    batch_embeddings.append(h.cpu().numpy())

            if batch_embeddings:
                embeddings.append(np.vstack(batch_embeddings))

            t.update(len(batch_smiles))

    molclr_features = np.vstack(embeddings) if embeddings else np.array([])
    
    return molclr_features