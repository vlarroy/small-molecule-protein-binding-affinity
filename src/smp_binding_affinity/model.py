import random
from typing import List
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch_geometric.nn import MessagePassing, global_max_pool
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import torch.optim as optim
import logging as log

from google.cloud import storage
from datatypes import Protein, Molecule


class _CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(_CustomGNNLayer, self).__init__(aggr='max')
        self.lin = nn.Linear(in_channels + 6, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return MessagePassing.propagate(self, edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        return combined

    def update(self, aggr_out):
        return self.lin(aggr_out)


class _GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(_GNNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([_CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
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

class SMPBindingAffinityModel:

    def __init__(
            self,
            gcs_bucket_name: str,
            gcs_model_path: str,
            binding_threshold = 0.8,
            train_test_size = 0.2,
            train_test_random_state = 42,
            batch_size = 32,
            data_loader_shuffle = True,
            train_hidden_dim = 64,
            train_num_epochs = 11,
            train_num_layers = 4,
            train_dropout_rate = 0.3,
            train_learning_rate = 0.001
        ):
        
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_model_path = gcs_model_path
        self.binding_threshold = binding_threshold
        self.train_test_size = train_test_size
        self.train_test_random_state = train_test_random_state
        self.batch_size = batch_size
        self.data_loader_shuffle = data_loader_shuffle
        self.train_hidden_dim = train_hidden_dim
        self.train_num_epochs = train_num_epochs
        self.train_num_layers = train_num_layers
        self.train_dropout_rate = train_dropout_rate
        self.train_learning_rate = train_learning_rate

    def _balance_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        # Get the count of binded and non-binded molecules
        binded_count = sum([molecule.is_binded for molecule in molecules])
        non_binded_count = len(molecules) - binded_count

        # Get the minimum count
        min_count = min(binded_count, non_binded_count)

        # Get a random sample of molecules with the minimum count
        if binded_count == min_count:
            molecules = [molecule for molecule in molecules if molecule.is_binded]
            molecules += random.sample([molecule for molecule in molecules if not molecule.is_binded], min_count)
        else:
            molecules = [molecule for molecule in molecules if not molecule.is_binded]
            molecules += random.sample([molecule for molecule in molecules if molecule.is_binded], min_count)

        # Shuffle the train molecules
        random.shuffle(molecules)

        return molecules

    def _one_hot_encoding(self, element, permitted_elements):
        """
        Maps input elements element which are not in the permitted list to the last element of the permitted list
        """
        if element not in permitted_elements:
            element = permitted_elements[-1]

        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: element==s , permitted_elements))]

        return binary_encoding

    def _get_bond_features(self, bond):

        bond_type = bond.GetBondType()

        features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
        ]
        
        return np.array(features, dtype=np.float32)

    def _get_atom_features(self, atom):
        
        #Define a simplified list of atom types
        permitted_atom_types = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
            'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
            'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
            'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
            'Pt', 'Hg', 'Pb', 'Dy', 'Unknown'
        ]
        atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
        atom_type_enc = self._one_hot_encoding(atom_type, permitted_atom_types)

        hybridization_type = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D
        ]
        atom_hybridization_type = self._one_hot_encoding(atom.GetHybridization(), hybridization_type)

        atom_degree = self._one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])

        is_in_ring = [int(atom.IsInRing())]

        total_hs = self._one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])

        implicit_valence = self._one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])

        chirality = self._one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        
        atom_features = atom_type_enc + atom_degree + is_in_ring + total_hs + implicit_valence + atom_hybridization_type + chirality
        
        return np.array(atom_features, dtype=np.float32)
    
    def _preprocess_data(self, molecules: List[Molecule]) -> List[Data]:

        id_list = [molecule.id for molecule in molecules]
        smile_list = [molecule.smile for molecule in molecules]

        labels_list = []
        for molecule in molecules:
            if molecule.is_binded is not None:
                labels_list.append(1 if molecule.is_binded else 0)
            else:
                labels_list.append(None)

        data_list = []

        for i in range(0, len(smile_list), self.batch_size):

            smiles_batch = smile_list[i:i+self.batch_size]
            molecules_batch = id_list[i:i+self.batch_size]
            labels_batch = None if labels_list.count(None) == len(labels_list) else labels_list[i:i+self.batch_size]

            for index, smile in enumerate(smiles_batch):
                mol = Chem.MolFromSmiles(smile)
                
                if not mol:  # Skip invalid SMILES strings
                    continue
                
                # Node features
                atom_features = [self._get_atom_features(atom) for atom in mol.GetAtoms()]
                x = torch.tensor(atom_features, dtype=torch.float)
                
                # Edge features
                edge_index = []
                edge_features = []
                for bond in mol.GetBonds():
                    bond_feature = self._get_bond_features(bond)
                    edge_features += [bond_feature, bond_feature]  # Same features in both directions

                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_index += [(start, end), (end, start)]  # Undirected graph
                
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
                
                # Creating the Data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data.molecule = molecules_batch[index]

                if labels_batch is not None:
                    data.y = torch.tensor([labels_batch[index]], dtype=torch.float)
                
                data_list.append(data)

        return data_list
    
    def _upload_model_to_gcs(self, local_path):
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket_name)
        blob = bucket.blob(self.gcs_model_path)
        blob.upload_from_filename(local_path)
        log.info(f"Model uploaded to GCS at gs://{self.gcs_bucket_name}/{self.gcs_model_path}")

    def _download_model_from_gcs(self, local_path):
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket_name)
        blob = bucket.blob(self.gcs_model_path)
        blob.download_to_filename(local_path)
        log.info(f"Model downloaded from GCS at gs://{self.gcs_bucket_name}/{self.gcs_model_path}")

    def predict(self, protein: Protein, molecules: List[Molecule], custom_binding_threshold: float = None) -> List[Molecule]:
        
        # Load the model from GCS
        model_path = f'{protein.acronym}_model.pt'
        self._download_model_from_gcs(model_path)
        model = torch.load(model_path)

        binding_threshold = self.binding_threshold
        if custom_binding_threshold is not None:
            binding_threshold = custom_binding_threshold

        data = self._preprocess_data(molecules)

        loader = DataLoader(data, batch_size=self.batch_size, shuffle=self.data_loader_shuffle)

        model.eval()
        predictions = []
        predicted_molecules = []

        with torch.no_grad():
            for data in loader:
                output = torch.sigmoid(model(data))
                predictions.extend(output.view(-1).tolist())
                predicted_molecules.extend(data.molecule)


        for i in range(len(predictions)):
            predicted_molecules[i].binding_affinity = predictions[i]
            predicted_molecules[i].is_binded = True if predictions[i] > binding_threshold else False

        return predicted_molecules
        
    
    def train(self, protein: Protein, molecules: List[Molecule]) -> None:

        balanced_molecules = self._balance_molecules(molecules)

        train_molecules, test_molecules = train_test_split(balanced_molecules, test_size=self.train_test_size, random_state=self.train_test_random_state)

        test_molecules_without_binding = map(lambda x: setattr(x, 'is_binded', None), copy.deepcopy(test_molecules))

        log.info(f"""Training on {len(train_molecules)} molecules, testing on {len(test_molecules)} molecules
            Configuration:
            - Binding threshold: {self.binding_threshold}
            - Train-test split: {self.train_test_size}
            - Random state: {self.train_test_random_state}
            - Batch size: {self.batch_size}
            - Hidden dim: {self.train_hidden_dim}
            - Num epochs: {self.train_num_epochs}
            - Num layers: {self.train_num_layers}
            - Dropout rate: {self.train_dropout_rate}
            - Learning rate: {self.train_learning_rate}
            - Data loader shuffle: {self.data_loader_shuffle}
            Other information:
            - Protein: {protein.acronym}
            - Molecules: {len(molecules)}
            - Train molecules: {len(train_molecules)} (binding: {sum([molecule.is_binded for molecule in train_molecules])}, non-binding: {len(train_molecules) - sum([molecule.is_binded for molecule in train_molecules])})
            - Test molecules without binding: {len(test_molecules_without_binding)}
            - Test molecules with binding: {len(test_molecules)}""")
        
        log.info('Starting preprocess...')
        train_data = self._preprocess_data(train_molecules)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.data_loader_shuffle)

        train_input_dim = train_loader.dataset[0].num_node_features

        log.info('Preprocess finished. Starting training...')
        model = _GNNModel(train_input_dim, self.train_hidden_dim, self.train_num_layers, self.train_dropout_rate)
        optimizer = optim.AdamW(model.parameters(), lr=self.train_learning_rate)
        criterion = BCEWithLogitsLoss()

        for epoch in range(self.train_num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.view(-1,1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            log.info(f'Epoch {epoch+1}/{self.train_num_epochs}, Loss: {total_loss / len(train_loader)}')
        
        # Save the model in GCS
        model_path = f'{protein.acronym}_model.pt'
        torch.save(model, model_path)
        self._upload_model_to_gcs(model_path)

        log.info('Training finished. Starting evaluation...')
        predicted_molecules = self.predict(protein, test_molecules_without_binding)

        log.info(classification_report(
            [predicted_molecules[i].is_binded for i in range(len(predicted_molecules))],
            [test_molecules[i].is_binded for i in range(len(test_molecules))]
        ))