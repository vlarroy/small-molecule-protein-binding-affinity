import numpy as np
import pandas as pd
import duckdb
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, global_max_pool
from torch.nn import BCEWithLogitsLoss
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List


def get_molecule_data_by_protein(file_path, protein, samples = None) -> pd.DataFrame:

    con = duckdb.connect()

    if samples:

        query = f"""
        (SELECT * FROM parquet_scan('{file_path}')
        WHERE binds = 0 AND protein_name = '{protein}'
        ORDER BY random()
        LIMIT {samples})
        UNION ALL
        (SELECT * FROM parquet_scan('{file_path}')
        WHERE binds = 1 AND protein_name = '{protein}'
        ORDER BY random()
        LIMIT {samples})
        """

    else:

        query = f"""
        SELECT * FROM parquet_scan('{file_path}')
        WHERE protein_name = '{protein}'
        """

    df = con.query(query).df()

    con.close()

    return df


def get_torch_data_object(smiles, ids, labels=None):

    def _one_hot_encoding(element, permitted_elements):
        """
        Maps input elements element which are not in the permitted list to the last element of the permitted list
        """
        if element not in permitted_elements:
            element = permitted_elements[-1]

        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: element==s , permitted_elements))]

        return binary_encoding


    def _get_atom_features(atom, use_chirality=True):
        
        #Define a simplified list of atom types
        permitted_atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Dy', 'Unknown']
        atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
        atom_type_enc = _one_hot_encoding(atom_type, permitted_atom_types)

        #Consider only the most impactful atom features: atom degree and whether the atom is in a ring
        atom_degree = _one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
        is_in_ring = [int(atom.IsInRing())]

        #Optionally include chirality
        if use_chirality:
            chirality_enc = _one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc
        else:
            atom_features = atom_type_enc + atom_degree + is_in_ring
        
        return np.array(atom_features, dtype=np.float32)


    def _get_bond_features(bond):

        #Simplified list of bond types
        permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, 'Unknown']
        bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'

        #Features: Bon type, is in a ring
        features = _one_hot_encoding(bond_type, permitted_bond_types) + [int(bond.IsInRing())]
        
        return np.array(features, dtype=np.float32)
    

    data_list = []
    
    for index, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        
        if not mol:  # Skip invalid SMILES strings
            continue
        
        # Node features
        atom_features = [_get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [(start, end), (end, start)]  # Undirected graph
            bond_feature = _get_bond_features(bond)
            edge_features += [bond_feature, bond_feature]  # Same features in both directions
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Creating the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.molecule_id = ids[index]

        if labels is not None:
            data.y = torch.tensor([labels[index]], dtype=torch.float)
        
        data_list.append(data)
    
    return data_list


def featurize_data_in_batches(ids_list, smiles_list, labels_list, batch_size = 2**8):
    data_list = []
    # Define tqdm progress bar
    pbar = tqdm(total=len(smiles_list), desc="Featurizing data")
    for i in range(0, len(smiles_list), batch_size):
        smiles_batch = smiles_list[i:i+batch_size]
        ids_batch = ids_list[i:i+batch_size]
        labels_batch = labels_list[i:i+batch_size] if labels_list is not None else None
        batch_data_list = get_torch_data_object(smiles_batch, ids_batch, labels_batch)
        data_list.extend(batch_data_list)
        pbar.update(len(smiles_batch))
        
    pbar.close()
    return data_list


def get_featurized_data(proteins: list, file_path: str, samples_per_protein: int = None):

    featurized_data = {}

    for protein in proteins:
        print(f"Processing {protein}...")

        df = get_molecule_data_by_protein(file_path, protein, samples_per_protein)

        smiles_list = df['molecule_smiles'].tolist()
        ids_list = df['id'].tolist()

        labels_list = None

        if 'binds' in df.columns:
            labels_list = df['binds'].tolist()

        featurized_data[protein] = featurize_data_in_batches(ids_list, smiles_list, labels_list)
    
    return featurized_data


class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels + 6, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return MessagePassing.propagate(self, edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        return combined

    def update(self, aggr_out):
        return self.lin(aggr_out)


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)


        x = global_max_pool(x, data.batch) # Global pooling to get a graph-level representation
        x = self.lin(x)
        return x

def train_model(
        loader,
        num_epochs,
        input_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        lr,
        save_path
    ):
    model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1,1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader)}')
    
    torch.save(model, save_path)

def predict_with_model(model_file_path, loader):
    print(f"Loading model {model_file_path}...")
    model = torch.load(model_file_path)


    model.eval()
    predictions = []
    molecule_ids = []

    with torch.no_grad():
        for data in loader:
            output = torch.sigmoid(model(data))
            predictions.extend(output.view(-1).tolist())
            molecule_ids.extend(data.molecule_id)
    
    return molecule_ids, predictions


def train(protein: str, data: list) -> None:
    
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=DATA_LOADER_SHUFFLE)

    train_input_dim = loader.dataset[0].num_node_features

    train_model(
        loader = loader,
        num_epochs = TRAIN_NUM_EPOCHS,
        input_dim = train_input_dim,
        hidden_dim = TRAIN_HIDDEN_DIM,
        num_layers = TRAIN_NUM_LAYERS,
        dropout_rate = TRAIN_DROPOUT_RATE,
        lr = TRAIN_LEARNING_RATE,
        save_path = f'{protein.lower()}_{MODEL_STATE_PATH}'
    )

def predict(protein: str, data: list) -> pd.DataFrame:

    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    molecule_ids, predictions = predict_with_model(f'{protein.lower()}_{MODEL_STATE_PATH}', loader)

    predictions = pd.DataFrame({
        'id': molecule_ids,
        'binds': predictions,
    }, index=None)

    predictions['id'] = predictions['id'].apply(lambda x: x.item())

    return predictions



TRAIN_PATH = 'train.parquet'
TEST_PATH = 'test.parquet'

SEH_TRAINED_PATH = 'seh_train_data.pt'
BRD4_TRAINED_PATH =  'brd4_train_data.pt'
HSA_TRAINED_PATH = 'hsa_train_data.pt'

SEH_TEST_PATH = 'seh_test_data.pt'
BRD4_TEST_PATH =  'brd4_test_data.pt'
HSA_TEST_PATH = 'hsa_test_data.pt'

MODEL_STATE_PATH = 'test_model.pt'

BINDING_THRESHOLD = 0.8

TRAIN_SAMPLES = 75000
TRAIN_TEST_SIZE = 0.2
TRAIN_TEST_RANDOM_STATE = 42
BATCH_SIZE = 32
DATA_LOADER_SHUFFLE = True
TRAIN_HIDDEN_DIM = 64
TRAIN_NUM_EPOCHS = 11
TRAIN_NUM_LAYERS = 4 
TRAIN_DROPOUT_RATE = 0.3
TRAIN_LEARNING_RATE = 0.001


proteins = ['HSA','sEH', 'BRD4']



featurized_data = get_featurized_data(proteins, TRAIN_PATH, samples_per_protein=TRAIN_SAMPLES)

train_test_split_dict = {}


for protein, data_list in featurized_data.items():

    train_data, test_data = train_test_split(data_list, test_size=TRAIN_TEST_SIZE, random_state=TRAIN_TEST_RANDOM_STATE)

    test_data_without_y = copy.deepcopy(test_data)
    for data in test_data_without_y:
        data.y = None

    train(protein, train_data)

    df = predict(protein, test_data_without_y)

    df['binds'] = df['binds'].apply(lambda x: 1 if x > BINDING_THRESHOLD else 0)

    print(f"{len(df['binds'])} - {len([data.y.item() for data in test_data])}")

    print(classification_report(df['binds'], [data.y.item() for data in test_data]))





# featurized_train_data = get_featurized_data(proteins, TRAIN_PATH, samples_per_protein=500)
# featurized_test_data = get_featurized_data(proteins, TEST_PATH)

# train(proteins, featurized_train_data)

# df = predict(proteins[0], featurized_test_data)