"""
Model definition for CellLineKG project.
Implements the GNN architecture based on KGDRP.
"""

import torch
import torch.nn as nn
import dgl.nn as dglnn
from dgl.nn import GraphConv, SAGEConv

class CellLineKGModel(nn.Module):
    """
    Main model for CellLineKG project.
    """
    def __init__(self, 
                 protein_dim, 
                 drug_dim, 
                 cell_line_dim, 
                 disease_dim,
                 hidden_dim=128,
                 num_layers=2):
        """
        Initialize the model.
        
        Args:
            protein_dim (int): Dimension of protein features.
            drug_dim (int): Dimension of drug features.
            cell_line_dim (int): Dimension of cell line features.
            disease_dim (int): Dimension of disease features.
            hidden_dim (int): Hidden dimension for GNN layers.
            num_layers (int): Number of GNN layers.
        """
        super(CellLineKGModel, self).__init__()
        
        # Feature embedding layers
        self.protein_embedding = nn.Linear(protein_dim, hidden_dim)
        self.drug_embedding = nn.Linear(drug_dim, hidden_dim)
        self.cell_line_embedding = nn.Linear(cell_line_dim, hidden_dim)
        self.disease_embedding = nn.Linear(disease_dim, hidden_dim)
        
        # GNN layers
        # Protein-Protein and Protein-Pathway edges use GraphSAGE
        self.sage_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))
        
        # CellLine-Protein edges use GCN
        self.gcn_layer = GraphConv(hidden_dim, hidden_dim)
        
        # MLP classifiers for auxiliary tasks
        self.rna_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dti_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.pathway_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Main task classifier
        self.drug_disease_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, graph, drug_features, protein_features, cell_line_features, disease_features):
        """
        Forward pass of the model.
        
        Args:
            graph (dgl.DGLHeteroGraph): Input heterogeneous graph.
            drug_features (torch.Tensor): Drug features.
            protein_features (torch.Tensor): Protein features.
            cell_line_features (torch.Tensor): Cell line features.
            disease_features (torch.Tensor): Disease features.
            
        Returns:
            dict: Dictionary containing outputs for all tasks.
        """
        # Embed features
        h_drug = self.drug_embedding(drug_features)
        h_protein = self.protein_embedding(protein_features)
        h_cell_line = self.cell_line_embedding(cell_line_features)
        h_disease = self.disease_embedding(disease_features)
        
        # Initialize node features dictionary
        h = {
            'drug': h_drug,
            'protein': h_protein,
            'cell_line': h_cell_line,
            'disease': h_disease
        }
        
        # GNN forward pass
        # For simplicity, we're showing a basic implementation
        # In practice, you would need to handle the heterogeneous graph properly
        for layer in self.sage_layers:
            h['protein'] = layer(graph, h['protein'])
        
        # Apply GCN layer for cell_line-protein edges
        h['protein'] = self.gcn_layer(graph, h['protein'])
        
        # Compute outputs for auxiliary tasks
        rna_output = self.rna_predictor(torch.cat([h['protein'], h['protein']], dim=1))
        dti_output = self.dti_predictor(torch.cat([h['drug'], h['protein']], dim=1))
        pathway_output = self.pathway_predictor(torch.cat([h['protein'], h['protein']], dim=1))
        
        # Compute output for main task
        drug_disease_output = self.drug_disease_predictor(torch.cat([h['drug'], h['disease']], dim=1))
        
        return {
            'rna': rna_output,
            'dti': dti_output,
            'pathway': pathway_output,
            'drug_disease': drug_disease_output
        }

if __name__ == "__main__":
    # Example usage
    print("Model definition ready.")