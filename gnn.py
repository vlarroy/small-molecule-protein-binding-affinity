# %% [markdown]
# # **GNNs for molecular representation and drug discovery**
# A natural approach to representing molecules is by using a graph structure, where atoms are depicted as nodes and bonds as edges. Various features, such as atom type, charge, and bond type, can be associated with these nodes and edges.
# 
# **Graph Neural Networks (GNNs) can be particularly useful for molecule classification tasks**. For instance, one intriguing application is predicting whether a given molecule (in this context, a small molecule) has the potential to act as an effective drug. This could be framed as a binary classification problem, determining whether a drug will bind to a specific protein.
# 
# To train the model, a curated dataset of compounds with known binding affinities can be used. Once trained, the model can then be applied to any molecule to predict its properties.
# 
# The trained model could be run on a large dataset of potential candidate molecules. From this, the top 100 molecules, as identified by the GNN model, could be selected for further detailed investigation by chemists.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb
import pickle
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch.nn import BCEWithLogitsLoss
from typing import Any
from sklearn.metrics import average_precision_score

# %%
train_path = 'train.parquet'
test_path = 'test.parquet'

con = duckdb.connect()

# %%
# Define the number of samples you want per category (binders and non-binders) per protein
samples_per_category = 1000  # Adjust this number as needed, this is just an example for the tutorial. 
#Your real values should be higher than this for proper training.

def get_balanced_data_for_protein(file_path, protein, samples):
    """
    Fetches a balanced dataset for a specific protein.
    
    Parameters:
    - file_path: Path to the dataset file.
    - protein: The name of the protein.
    - samples: Number of samples per binder category (1 or 0).
    
    Returns:
    - A pandas DataFrame containing the balanced dataset for the protein.
    """
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
    return con.query(query).df()

# List of proteins to query for
proteins = ['sEH', 'BRD4', 'HSA']

# Creating a dictionary of dataframes, one for each protein
datasets = {}
for protein in proteins:
    datasets[protein] = get_balanced_data_for_protein(train_path, protein, samples_per_category)

# At this point, `datasets` contains separate dataframes for sEH, BRD4, and HSA
# For example, to access the dataset for sEH:
seh_df = datasets['sEH']

# And similarly for BRD4 and HSA
brd4_df = datasets['BRD4']
hsa_df = datasets['HSA']


'''We now have balanced data for training for each protein, 
which we will featurize for our GNN model'''

# %%
'''We do a similar process to load in the test datasets, 
however this time we only filter for the proteins, and
we don't balance them'''
def get_data_for_protein(file_path, protein):
    """
    Fetches the dataset for a specific protein from the test dataset.
    
    Parameters:
    - file_path: Path to the dataset file.
    - protein: The name of the protein.
    
    Returns:
    - A pandas DataFrame containing the dataset for the protein.
    """
    query = f"""
    SELECT * FROM parquet_scan('{file_path}')
    WHERE protein_name = '{protein}'
    """
    return con.query(query).df()


# Assuming the connection `con` to DuckDB is still open from the previous operations

# Creating a dictionary to store the filtered test datasets
test_datasets = {}
for protein in proteins:  # Using the same list of proteins: ['sEH', 'BRD4', 'HSA']
    test_datasets[protein] = get_data_for_protein(test_path, protein)

# At this point, `test_datasets` contains separate dataframes for sEH, BRD4, and HSA without balancing
# For example, to access the test dataset for sEH:
seh_test_df = test_datasets['sEH']

# And similarly for BRD4 and HSA
brd4_test_df = test_datasets['BRD4']
hsa_test_df = test_datasets['HSA']
con.close()

# %%
seh_df.head()

# %%
seh_test_df.head() #note we don't have a binds column here

# %% [markdown]
# ## **Featurizing**
# 
# For this step, we are going to use several methods from the 'rdkit' library. Below a simple explanation of each:
# - **.GetSymbol()**: Returns the atomic symbol (a string).
# 
# - **.GetDegree()**: Returns the degree of the atom in the molecule, which is defined to be its number of directly-bonded neighbors. The degree is independent of bond orders, but is dependent on whether or not Hs are explicit in the graph.
# 
# - **.IsInRing()**: Returns whether or not the atom is in a ring.
# 
# - **.GetChiralTag()**: Retreives the chiral information of an atom in a molecule. Chirality is a geometric property of some molecules and ions, where a molecule/ion is not superimposable on its mirror image. The chirality of a molecule is important because it can drasticallly influence its biological activity and interactions.
# 
# - **.GetAtoms()**: Returns an iterator over the atoms in a molecule.
# 
# - **.GetBondType()**: Returns the type of the bond as BondType
# 
# - **.GetBeginAtomIdx()**: Returns the index of atom where the bond starts.
# 
# - **.GetEndAtomIdx()**: Returns the index of the atom where the bond ends.
# 
# For more info, please check [rdkit.Chem.rdchem module](https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Atom.GetChiralTag)

# %%
#Auxiliary function for one-hot encoding transformation based on list of permitted values:
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element of the permitted list
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x==s , permitted_list))]
    return binary_encoding

# %% [markdown]
# #### **1. Atom featurization**

# %%
#Atom featurization function
def get_atom_features(atom, use_chirality=True):
    #Define a simplified list of atom types
    permitted_atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Dy', 'Unknown']
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
    atom_type_enc = one_hot_encoding(atom_type, permitted_atom_types)

    #Consider only the most impactful atom features: atom degree and whether the atom is in a ring
    atom_degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
    is_in_ring = [int(atom.IsInRing())]

    #Optionally include chirality
    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc
    else:
        atom_features = atom_type_enc + atom_degree + is_in_ring
    
    return np.array(atom_features, dtype=np.float32)

# %% [markdown]
# #### **2. Bond featurization**
# (if this works, we can try then to get more information such as bond order, bond stereochemistry, etc.)

# %%
def get_bond_features(bond):
    #Simplified list of bond types
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, 'Unknown']
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'

    #Features: Bon type, is in a ring
    features = one_hot_encoding(bond_type, permitted_bond_types) \
               + [int(bond.IsInRing())]
    
    return np.array(features, dtype=np.float32)

# %% [markdown]
# ### **Converting the molecular structure into graph representation suitable for GNN.**
# 
# - The **nodes** represent the atoms in the molecule.
# - The **edges** represent the bonds between atoms.
# - The **edge_index** specifies the connectivity of the graph (which atoms are connected by bonds)
# - The **edge_features** provides additional infromation about the bonds (bond type or properties)

# %%
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, ids, y=None):
    data_list = []
    
    for index, smiles in enumerate(x_smiles):
        mol = Chem.MolFromSmiles(smiles)
        
        if not mol:  # Skip invalid SMILES strings
            continue
        
        # Node features
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [(start, end), (end, start)]  # Undirected graph
            bond_feature = get_bond_features(bond)
            edge_features += [bond_feature, bond_feature]  # Same features in both directions
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Creating the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.molecule_id = ids[index]
        if y is not None:
            data.y = torch.tensor([y[index]], dtype=torch.float)
        
        data_list.append(data)
    
    return data_list

# %%
def featurize_data_in_batches(smiles_list, labels_list, batch_size):
    data_list = []
    # Define tqdm progress bar
    pbar = tqdm(total=len(smiles_list), desc="Featurizing data")
    for i in range(0, len(smiles_list), batch_size):
        smiles_batch = smiles_list[i:i+batch_size]
        labels_batch = labels_list[i:i+batch_size]
        ids_batch = ids_list[i:i+batch_size]
        batch_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_batch, ids_batch, labels_batch)
        data_list.extend(batch_data_list)
        pbar.update(len(smiles_batch))
        
    pbar.close()
    return data_list

# %%
# Define the batch size for featurization
batch_size = 2**8
# List of proteins and their corresponding dataframes
proteins_data = {
    'sEH': seh_df,
    'BRD4': brd4_df,
    'HSA': hsa_df
}
# Dictionary to store the featurized data for each protein
featurized_data = {}
# Loop over each protein and its dataframe
for protein_name, df in proteins_data.items():
    print(f"Processing {protein_name}...")
    smiles_list = df['molecule_smiles'].tolist()
    ids_list = df['id'].tolist()
    labels_list = df['binds'].tolist()
 # Featurize the data
    featurized_data[protein_name] = featurize_data_in_batches(smiles_list, labels_list, batch_size)
    

seh_train_data = featurized_data['sEH']
brd4_train_data = featurized_data['BRD4']
hsa_train_data = featurized_data['HSA']

# %%
torch.save(seh_train_data, 'seh_train_data.pt')
torch.save(brd4_train_data, 'brd4_train_data.pt')
torch.save(hsa_train_data, 'hsa_train_data.pt')

# %% [markdown]
# ### Test data featurizing process

# %%
batch_size = 2**8
smiles_list = brd4_test_df['molecule_smiles'].tolist()
ids_list = brd4_test_df['id'].tolist()
labels_list = [-1]*len(smiles_list) #we dont have the actual labels, so we assign some dummy label list for the function. (don't choose 0 or 1)
brd4_test_data = featurize_data_in_batches(smiles_list, labels_list, batch_size)

batch_size = 2**8
smiles_list  = seh_test_df['molecule_smiles'].tolist()
ids_list = seh_test_df['id'].tolist()
labels_list = [-1]*len(smiles_list)
seh_test_data = featurize_data_in_batches(smiles_list, labels_list, batch_size)

batch_size = 2**8
smiles_list  = hsa_test_df['molecule_smiles'].tolist()
ids_list = hsa_test_df['id'].tolist()
labels_list = [-1]*len(smiles_list)
hsa_test_data = featurize_data_in_batches(smiles_list, labels_list, batch_size)

# %%
seh_test_data[0] #example of what a pytorch object looks like

# %%
torch.save(seh_test_data, 'seh_test_data.pt')
torch.save(brd4_test_data, 'brd4_test_data.pt')
torch.save(hsa_test_data, 'hsa_test_data.pt')

# %% [markdown]
# ## **Training and test**

# %%
featurized_data_train = {
    'sEH': seh_train_data,
    'BRD4': brd4_train_data,
    'HSA': hsa_train_data
}

featurized_data_test = {
    'sEH': seh_test_data,
    'BRD4': brd4_test_data,
    'HSA': hsa_test_data
}

# %% [markdown]
# ### Define custom GNN layer

# %%
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

# %% [markdown]
# ### Define GNN model

# %%
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

# %%
def train_model(train_loader, num_epochs, input_dim, hidden_dim, num_layers, dropout_rate, lr):
    model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1,1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')
    return model

def predict_with_model(model, test_loader):
    model.eval()
    predictions = []
    molecule_ids = []

    with torch.no_grad():
        for data in test_loader:
            output = torch.sigmoid(model(data))
            predictions.extend(output.view(-1).tolist())
            molecule_ids.extend(data.molecule_id)
    
    return molecule_ids, predictions

# %%
proteins = ['sEH', 'BRD4', 'HSA']
all_predictions = []

for protein in proteins:
    print(f"Training and predicting for {protein}")
    
    # Create DataLoaders for the current protein
    train_loader = DataLoader(featurized_data_train[protein], batch_size=32, shuffle=True)
    test_loader = DataLoader(featurized_data_test[protein], batch_size=32, shuffle=False)
    
    # Train model
    input_dim = train_loader.dataset[0].num_node_features
    hidden_dim = 64
    num_epochs = 11
    num_layers = 4 #Should ideally be set so that all nodes can communicate with each other
    dropout_rate = 0.3
    lr = 0.001
    #These are just example values, feel free to play around with them.
    model = train_model(train_loader,num_epochs, input_dim, hidden_dim,num_layers, dropout_rate, lr)
    
    # Predict
    molecule_ids, predictions = predict_with_model(model, test_loader)
    
    # Collect predictions
    protein_predictions = pd.DataFrame({
        'id': molecule_ids,
        'binds': predictions,
    })
    all_predictions.append(protein_predictions)

# %%
#Combine all predictions into one DataFrame
final_predictions = pd.concat(all_predictions, ignore_index=True)
#Convert 'molecule_id' from tensors to integers directly within the DataFrame
final_predictions['id'] = final_predictions['id'].apply(lambda x: x.item())

#Save the modified DataFrame to a CSV file
final_predictions.to_csv('final_predictions', index=False)

# %%
final_predictions.head()


