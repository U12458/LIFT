import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

def smiles_to_graph(smiles, device):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_features = []
    for atom in mol.GetAtoms():
        atom_type = ATOM_LIST.index(atom.GetAtomicNum())
        chirality_tag = CHIRALITY_LIST.index(atom.GetChiralTag())
        atom_features.append([atom_type, chirality_tag])

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_LIST.index(bond.GetBondType())
        bond_direction = BONDDIR_LIST.index(bond.GetBondDir())
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([bond_type, bond_direction])
        edge_attr.append([bond_type, bond_direction])

    atom_features = torch.tensor(atom_features, dtype=torch.long).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long).to(device)
    batch = torch.zeros(atom_features.size(0), dtype=torch.long).to(device)

    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=torch.tensor([0.0]))

def smiles_list_to_graph(smiles_list, device):
    data_list = []
    for smiles in smiles_list:
        try:
            data_list.append(smiles_to_graph(smiles, device))
        except ValueError as e:
            print(f"Skipping invalid SMILES: {smiles}. Error: {e}")
    
    return data_list