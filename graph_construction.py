"""
Graph construction module for CellLineKG project.
Builds the CellLineBioHG heterogeneous graph using DGL.
"""

import dgl
import torch
import numpy as np
from config import PPI_BUILD_METHOD, PPI_CORR_THRESHOLD, PPI_PERTURB_THRESHOLD, EXPR_Z_THRESHOLD

def create_cell_line_bio_hg(proteins, drugs, cell_lines, diseases, 
                           drug_protein_edges, protein_protein_edges, 
                           cell_line_protein_edges, disease_protein_edges,
                           disease_cell_line_edges):
    """
    Create the CellLineBioHG heterogeneous graph.
    
    Args:
        proteins (list): List of protein nodes.
        drugs (list): List of drug nodes.
        cell_lines (list): List of cell line nodes.
        diseases (list): List of disease nodes.
        drug_protein_edges (list of tuples): (drug, protein) edges.
        protein_protein_edges (list of tuples): (protein, protein) edges.
        cell_line_protein_edges (list of tuples): (cell_line, protein) edges with weights.
        disease_protein_edges (list of tuples): (disease, protein) edges.
        disease_cell_line_edges (list of tuples): (disease, cell_line) edges.
        
    Returns:
        dgl.DGLHeteroGraph: Constructed heterogeneous graph.
    """
    # Create node dictionaries
    protein_ids = {p: i for i, p in enumerate(proteins)}
    drug_ids = {d: i for i, d in enumerate(drugs)}
    cell_line_ids = {c: i for i, c in enumerate(cell_lines)}
    disease_ids = {d: i for i, d in enumerate(diseases)}
    
    # Create graph
    hg = dgl.heterograph({
        ('drug', 'drug_protein', 'protein'): (
            [drug_ids[d] for d, p in drug_protein_edges],
            [protein_ids[p] for d, p in drug_protein_edges]
        ),
        ('protein', 'protein_protein', 'protein'): (
            [protein_ids[p1] for p1, p2 in protein_protein_edges],
            [protein_ids[p2] for p1, p2 in protein_protein_edges]
        ),
        ('cell_line', 'cell_line_protein', 'protein'): (
            [cell_line_ids[c] for c, p, _ in cell_line_protein_edges],
            [protein_ids[p] for c, p, _ in cell_line_protein_edges]
        ),
        ('disease', 'disease_protein', 'protein'): (
            [disease_ids[d] for d, p in disease_protein_edges],
            [protein_ids[p] for d, p in disease_protein_edges]
        ),
        ('disease', 'disease_cell_line', 'cell_line'): (
            [disease_ids[d] for d, c in disease_cell_line_edges],
            [cell_line_ids[c] for d, c in disease_cell_line_edges]
        )
    })
    
    # Add edge weights for cell_line_protein edges (for sampling only)
    # Note: These weights are not used in GNN message passing
    clp_weights = torch.FloatTensor([w for _, _, w in cell_line_protein_edges])
    hg.edges['cell_line_protein'].data['weight'] = clp_weights
    
    return hg

def build_protein_protein_edges(data_processor):
    """
    Build protein-protein edges based on configured method.
    
    Args:
        data_processor: Data processor object with methods to compute edges.
        
    Returns:
        list of tuples: Protein-protein edges.
    """
    if PPI_BUILD_METHOD == "coexpression":
        # Placeholder for coexpression-based edges
        return []
    elif PPI_BUILD_METHOD == "perturbation":
        # Use perturbation response similarity
        similarity_df = data_processor.compute_perturbation_response_similarity()
        edges = []
        for i in range(len(similarity_df.columns)):
            for j in range(i+1, len(similarity_df.columns)):
                if similarity_df.iloc[i, j] > PPI_PERTURB_THRESHOLD:
                    edges.append((similarity_df.columns[i], similarity_df.columns[j]))
        return edges
    else:
        raise ValueError(f"Unknown PPI build method: {PPI_BUILD_METHOD}")

if __name__ == "__main__":
    # Example usage
    print("Graph construction module ready.")