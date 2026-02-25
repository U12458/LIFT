import torch
from torch.utils.data import Dataset

class LNPDataset(Dataset):
    def __init__(self, X_ecfp, X_chemberta, X_molclr, X_extra, y):
        
        self.X_ecfp = X_ecfp
        self.X_chemberta = X_chemberta
        self.X_molclr = X_molclr
        
        self.X_extra = X_extra
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_ecfp[idx], self.X_chemberta[idx], self.X_molclr[idx],
            self.X_extra[idx], self.y[idx]
        )