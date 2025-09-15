"""
Data preprocessing module for CellLineKG project.
Handles downloading and preprocessing of Tahoe-100M, Enrichr, and DTI data.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from config import MIN_CELL_LINES_FOR_TOP80, DISEASE_PROTEIN_TOP_K

def load_tahoe_data(gcs_path):
    """
    Load Tahoe-100M data from Google Cloud Storage.
    
    Args:
        gcs_path (str): Path to the Tahoe h5ad file in GCS.
        
    Returns:
        AnnData: Loaded AnnData object.
    """
    import fsspec
    fs = fsspec.filesystem('gs')
    with fs.open(gcs_path, 'rb') as f:
        adata = sc.read_h5ad(f)
    return adata

def filter_proteins_by_expression(adata, min_cell_lines=MIN_CELL_LINES_FOR_TOP80):
    """
    Filter proteins (genes) based on expression levels across cell lines.
    
    Args:
        adata (AnnData): Tahoe data.
        min_cell_lines (int): Minimum number of cell lines where gene expression
                              should be in top 80%.
                              
    Returns:
        list: List of filtered gene names (Uniprot IDs).
    """
    # This is a placeholder implementation
    # In practice, you would implement the filtering logic here
    # For now, we'll just return all genes
    return adata.var_names.tolist()

def compute_perturbation_response_similarity(adata):
    """
    Compute perturbation response similarity between genes.
    
    Args:
        adata (AnnData): Tahoe data.
        
    Returns:
        pd.DataFrame: DataFrame with gene-gene similarities.
    """
    # Placeholder implementation
    # Actual implementation would compute similarities based on perturbation responses
    genes = adata.var_names.tolist()
    similarity_df = pd.DataFrame(np.random.rand(len(genes), len(genes)), 
                                index=genes, columns=genes)
    return similarity_df

def get_disease_genes(disease_name, top_k=DISEASE_PROTEIN_TOP_K):
    """
    Get disease-related genes from Enrichr.
    
    Args:
        disease_name (str): Name of the disease.
        top_k (int): Number of top genes to retrieve.
        
    Returns:
        list: List of disease-related genes.
    """
    # Placeholder implementation
    # Actual implementation would query Enrichr API
    return [f"GENE_{i}" for i in range(top_k)]

if __name__ == "__main__":
    # Example usage
    tahoe_path = "gs://arc-ctc-tahoe100/2025-02-25/tutorial/plate3_2k-obs.h5ad"
    adata = load_tahoe_data(tahoe_path)
    filtered_genes = filter_proteins_by_expression(adata)
    print(f"Filtered genes: {len(filtered_genes)}")