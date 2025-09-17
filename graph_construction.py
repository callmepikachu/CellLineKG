"""
Graph construction module for CellLineKG project.
Builds the CellLineBioHG heterogeneous graph using DGL.
"""

import dgl
import torch
import numpy as np
from config import PPI_BUILD_METHOD, PPI_CORR_THRESHOLD, PPI_PERTURB_THRESHOLD, EXPR_Z_THRESHOLD


# File: graph_construction.py (替换原函数)

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
        cell_line_protein_edges (list of tuples): (cell_line, protein, weight) edges.
        disease_protein_edges (list of tuples): (disease, protein) edges.
        disease_cell_line_edges (list of tuples): (disease, cell_line) edges.

    Returns:
        dgl.DGLHeteroGraph: Constructed heterogeneous graph.
    """
    # Create node dictionaries with error checking
    protein_ids = {p: i for i, p in enumerate(proteins)}
    drug_ids = {d: i for i, d in enumerate(drugs)}
    cell_line_ids = {c: i for i, c in enumerate(cell_lines)}
    disease_ids = {d: i for i, d in enumerate(diseases)}

    # Prepare edge index lists
    def get_edge_indices(edge_list, src_dict, dst_dict, edge_type):
        src_idx, dst_idx = [], []
        valid_edges = []
        for edge in edge_list:
            try:
                if len(edge) == 2:
                    src, dst = edge
                    src_idx.append(src_dict[src])
                    dst_idx.append(dst_dict[dst])
                    valid_edges.append(edge)
                elif len(edge) == 3:  # for cell_line_protein with weight
                    src, dst, _ = edge
                    src_idx.append(src_dict[src])
                    dst_idx.append(dst_dict[dst])
                    valid_edges.append(edge)
            except KeyError as e:
                print(f"Warning: Skipping edge {edge} in {edge_type} due to missing node: {e}")
                continue
        return (src_idx, dst_idx), valid_edges

    # Build edge index tuples
    dp_edges, _ = get_edge_indices(drug_protein_edges, drug_ids, protein_ids, 'drug_protein')
    pp_edges, _ = get_edge_indices(protein_protein_edges, protein_ids, protein_ids, 'protein_protein')
    clp_edges, clp_edges_with_weight = get_edge_indices(cell_line_protein_edges, cell_line_ids, protein_ids,
                                                        'cell_line_protein')
    dpe_edges, _ = get_edge_indices(disease_protein_edges, disease_ids, protein_ids, 'disease_protein')
    dcl_edges, _ = get_edge_indices(disease_cell_line_edges, disease_ids, cell_line_ids, 'disease_cell_line')

    # Create graph
    hg = dgl.heterograph({
        ('drug', 'drug_protein', 'protein'): dp_edges,
        ('protein', 'protein_protein', 'protein'): pp_edges,
        ('cell_line', 'cell_line_protein', 'protein'): clp_edges,
        ('disease', 'disease_protein', 'protein'): dpe_edges,
        ('disease', 'disease_cell_line', 'cell_line'): dcl_edges
    })

    # Add edge weights for cell_line_protein edges (for sampling only)
    if clp_edges_with_weight:
        clp_weights = torch.FloatTensor([w for _, _, w in clp_edges_with_weight])
        hg.edges['cell_line_protein'].data['weight'] = clp_weights

    print(f"Graph built with {hg.number_of_nodes()} nodes and {hg.number_of_edges()} edges.")
    return hg



def build_protein_protein_edges(adata, gene_list):
    """
    Build protein-protein edges based on configured method.

    Args:
        adata (AnnData): Tahoe data.
        gene_list (list): List of filtered gene names.

    Returns:
        list of tuples: Protein-protein edges.
    """
    from config import PPI_BUILD_METHOD, PPI_CORR_THRESHOLD, PPI_PERTURB_THRESHOLD
    from data_preprocessing import compute_coexpression_ppi, compute_perturbation_response_similarity

    if PPI_BUILD_METHOD == "coexpression":
        return compute_coexpression_ppi(adata, gene_list, threshold=PPI_CORR_THRESHOLD)
    elif PPI_BUILD_METHOD == "perturbation":
        return compute_perturbation_response_similarity(adata, gene_list, threshold=PPI_PERTURB_THRESHOLD)
    else:
        raise ValueError(f"Unknown PPI build method: {PPI_BUILD_METHOD}")

if __name__ == "__main__":
    # Example usage
    print("Graph construction module ready.")