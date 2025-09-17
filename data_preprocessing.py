"""
Data preprocessing module for CellLineKG project.
Handles downloading and preprocessing of Tahoe-100M, Enrichr, and DTI data.
"""
from random import random
import time
import requests
import scanpy as sc
import numpy as np

import fsspec
import scanpy as sc

import pandas as pd
from config import  PPI_PERTURB_THRESHOLD, PPI_CORR_THRESHOLD, DISEASE_PROTEIN_TOP_K, MIN_CELL_LINES_FOR_TOP80


def load_tahoe_data(gcs_path, n_cells=None):

    fs = fsspec.filesystem('gs')
    with fs.open(gcs_path, 'rb') as f:
        adata = sc.read_h5ad(f)

    if n_cells is not None and n_cells < adata.n_obs:
        print(f"⚠️  [DEBUG] Subsetting to first {n_cells} cells.")
        adata = adata[:n_cells, :].copy()

    print(f"✅ Loaded data: {adata.n_obs} cells, {adata.n_vars} genes.")
    return adata

# File: data_preprocessing.py
# 替换原函数

def filter_proteins_by_expression(adata, min_cell_lines=MIN_CELL_LINES_FOR_TOP80):
    """
    Filter proteins (genes) based on expression levels across cell lines.
    Keep genes that are in the top 80% expression in at least `min_cell_lines` cell lines.
    (即：在至少 N 个细胞系中，该基因的表达量 > 该细胞系所有基因的80百分位数)

    Args:
        adata (AnnData): Tahoe data.
        min_cell_lines (int): Minimum number of cell lines where gene expression should be in top 80%.

    Returns:
        list: List of filtered gene names (as strings).
    """
    # Convert to dense array if sparse
    if hasattr(adata.X, "toarray"):
        expr_matrix = adata.X.toarray()  # shape: (n_cells, n_genes)
    else:
        expr_matrix = adata.X

    gene_names = np.array(adata.var_names.tolist())  # 转为numpy array方便索引
    n_cells, n_genes = expr_matrix.shape

    # 初始化计数器
    gene_count = np.zeros(n_genes, dtype=int)

    # 对每个细胞（cell line），计算其80th百分位数阈值
    for i in range(n_cells):
        cell_expr = expr_matrix[i, :]
        threshold = np.percentile(cell_expr, 80)  # 80th percentile
        # 找出在这个细胞中表达量 > 阈值的基因
        top_genes_mask = cell_expr > threshold
        gene_count[top_genes_mask] += 1

    # 筛选在至少 min_cell_lines 个细胞中进入 top 80% 的基因
    filtered_mask = gene_count >= min_cell_lines
    filtered_genes = gene_names[filtered_mask].tolist()

    print(f"Filtered from {n_genes} to {len(filtered_genes)} genes.")
    return filtered_genes

def compute_perturbation_response_similarity(adata, gene_list, drug_col='drug', cell_col='cell_name',
                                              reference_drug='DMSO', threshold=PPI_PERTURB_THRESHOLD):
    """
    Compute perturbation response similarity between genes.
    Uses log2 fold change vs reference (e.g., DMSO) as response vector.

    Args:
        adata (AnnData): Tahoe data.
        gene_list (list): List of gene names to consider.
        drug_col (str): Column name in adata.obs for drug perturbation.
        cell_col (str): Column name in adata.obs for cell line.
        reference_drug (str): Reference condition (e.g., 'DMSO').
        threshold (float): Cosine similarity threshold to create an edge.

    Returns:
        list of tuples: (gene1, gene2) edges where cosine similarity >= threshold.
    """
    # Subset to selected genes
    adata_sub = adata[:, gene_list].copy()

    # Get unique cell lines
    cell_lines = adata_sub.obs[cell_col].unique()

    gene_response_dict = {gene: [] for gene in gene_list}

    for cell_line in cell_lines:
        # 获取参照组表达
        ref_mask = (adata_sub.obs[drug_col] == reference_drug) & (adata_sub.obs[cell_col] == cell_line)
        if ref_mask.sum() == 0:
            continue
        ref_expr = adata_sub[ref_mask, :].X
        if hasattr(ref_expr, "toarray"):
            ref_expr = ref_expr.toarray()
        ref_mean = np.mean(ref_expr, axis=0)  # shape: (n_genes,)

        # 获取所有处理组
        treat_drugs = adata_sub.obs[drug_col].unique()
        for drug in treat_drugs:
            if drug == reference_drug:
                continue
            treat_mask = (adata_sub.obs[drug_col] == drug) & (adata_sub.obs[cell_col] == cell_line)
            if treat_mask.sum() == 0:
                continue

            treat_expr = adata_sub[treat_mask, :].X
            if hasattr(treat_expr, "toarray"):
                treat_expr = treat_expr.toarray()
            treat_mean = np.mean(treat_expr, axis=0)  # shape: (n_genes,)

            # 计算 log2FC
            log2fc = np.log2(treat_mean + 1) - np.log2(ref_mean + 1)

            for i, gene in enumerate(gene_list):
                gene_response_dict[gene].append(log2fc[i])

    # Convert to matrix: genes x perturbations
    response_matrix = np.array([gene_response_dict[gene] for gene in gene_list])  # shape: (n_genes, n_perturbations)

    # Normalize each gene vector to unit length for cosine similarity
    norms = np.linalg.norm(response_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    response_matrix_normalized = response_matrix / norms

    # Compute cosine similarity matrix
    cosine_sim = np.dot(response_matrix_normalized, response_matrix_normalized.T)  # shape: (n_genes, n_genes)

    # Extract edges
    edges = []
    n_genes = len(gene_list)
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if cosine_sim[i, j] >= threshold:
                edges.append((gene_list[i], gene_list[j]))

    print(f"Built {len(edges)} perturbation-response PPI edges (threshold={threshold}).")
    return edges

def get_disease_genes(disease_name, top_k=DISEASE_PROTEIN_TOP_K):
    """
    Get disease-related genes from Enrichr.

    Args:
        disease_name (str): Name of the disease.
        top_k (int): Number of top genes to retrieve.

    Returns:
        list: List of disease-related genes (gene symbols).
    """
    # 优先从 DisGeNET 获取，若为空则尝试 GWAS Catalog
    genes = query_enrichr_gene_set(disease_name, "DisGeNET")
    if len(genes) == 0:
        genes = query_enrichr_gene_set(disease_name, "GWAS_Catalog")

    # 截取 top_k
    return genes[:top_k] if len(genes) > top_k else genes


def compute_coexpression_ppi(adata, gene_list, threshold=PPI_CORR_THRESHOLD):
    """
    Compute protein-protein interaction (PPI) edges based on gene co-expression across cell lines.

    Args:
        adata (AnnData): Tahoe data.
        gene_list (list): List of gene names to consider.
        threshold (float): Pearson correlation threshold to create an edge.

    Returns:
        list of tuples: (gene1, gene2) edges where correlation >= threshold.
    """
    # Subset adata to selected genes
    adata_sub = adata[:, gene_list]

    # Convert to dense
    if hasattr(adata_sub.X, "toarray"):
        expr_matrix = adata_sub.X.toarray().T  # shape: (n_genes, n_cells)
    else:
        expr_matrix = adata_sub.X.T

    n_genes = len(gene_list)
    edges = []

    # Compute pairwise Pearson correlation
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            gene_i_expr = expr_matrix[i, :]
            gene_j_expr = expr_matrix[j, :]

            # Skip if constant (avoid NaN)
            if np.std(gene_i_expr) == 0 or np.std(gene_j_expr) == 0:
                continue

            corr = np.corrcoef(gene_i_expr, gene_j_expr)[0, 1]
            if corr >= threshold:
                edges.append((gene_list[i], gene_list[j]))

    print(f"Built {len(edges)} co-expression PPI edges (threshold={threshold}).")
    return edges


# File: data_preprocessing.py (追加)

def build_disease_cell_line_edges(disease_names, cell_line_names, adata, method="de"):
    """
    构建 Disease-CellLine 边（真实逻辑）。

    Args:
        disease_names (list): 疾病名列表
        cell_line_names (list): 细胞系名列表
        adata (AnnData): Tahoe 数据，用于获取基因表达
        method (str): "de" 或 "gwas"

    Returns:
        list of tuples: (disease, cell_line) 边列表
    """
    edges = []

    if method == "de":
        # 策略：疾病相关基因在某个细胞系中高表达 → 连接
        for disease in disease_names:
            disease_genes = get_disease_genes(disease)
            if len(disease_genes) == 0:
                continue

            # 获取这些基因在 adata 中的索引
            gene_mask = adata.var_names.isin(disease_genes)
            if gene_mask.sum() == 0:
                continue

            # 对每个细胞系，计算疾病基因的平均表达量
            for cell_line in cell_line_names:
                cell_mask = adata.obs['cell_name'] == cell_line
                if cell_mask.sum() == 0:
                    continue

                # 计算该细胞系中疾病基因的平均表达
                expr_subset = adata[cell_mask, gene_mask].X
                if hasattr(expr_subset, "toarray"):
                    expr_subset = expr_subset.toarray()
                mean_expr = np.mean(expr_subset)

                # 如果平均表达 > 1.0 (CPM 或 TPM 单位)，则连接
                if mean_expr > 1.0:
                    edges.append((disease, cell_line))
                    print(f"Connected {disease} → {cell_line} via DE (mean expr={mean_expr:.2f})")

    elif method == "gwas":
        # 策略：GWAS 基因在细胞系中高表达 → 连接
        for disease in disease_names:
            # 从 GWAS Catalog 获取基因
            gwas_genes = query_enrichr_gene_set(disease, "GWAS_Catalog")
            if len(gwas_genes) == 0:
                continue

            gene_mask = adata.var_names.isin(gwas_genes)
            if gene_mask.sum() == 0:
                continue

            for cell_line in cell_line_names:
                cell_mask = adata.obs['cell_name'] == cell_line
                if cell_mask.sum() == 0:
                    continue

                expr_subset = adata[cell_mask, gene_mask].X
                if hasattr(expr_subset, "toarray"):
                    expr_subset = expr_subset.toarray()
                mean_expr = np.mean(expr_subset)

                if mean_expr > 1.0:
                    edges.append((disease, cell_line))
                    print(f"Connected {disease} → {cell_line} via GWAS (mean expr={mean_expr:.2f})")

    print(f"Built {len(edges)} Disease-CellLine edges using method '{method}'.")
    return edges


def filter_proteins_by_expression(adata, min_cell_lines=MIN_CELL_LINES_FOR_TOP80):
    """
    Filter proteins (genes) based on expression levels across cell lines.
    Keep genes that are in the top 80% expression in at least `min_cell_lines` cell lines.
    (即：在至少 N 个细胞系中，该基因的表达量 > 该细胞系所有基因的80百分位数)

    Args:
        adata (AnnData): Tahoe data.
        min_cell_lines (int): Minimum number of cell lines where gene expression should be in top 80%.

    Returns:
        list: List of filtered gene names (as strings).
    """
    # Convert to dense array if sparse
    if hasattr(adata.X, "toarray"):
        expr_matrix = adata.X.toarray()  # shape: (n_cells, n_genes)
    else:
        expr_matrix = adata.X

    gene_names = np.array(adata.var_names.tolist())  # 转为numpy array方便索引
    n_cells, n_genes = expr_matrix.shape

    # 初始化计数器
    gene_count = np.zeros(n_genes, dtype=int)

    # 对每个细胞（cell line），计算其80th百分位数阈值
    for i in range(n_cells):
        cell_expr = expr_matrix[i, :]
        threshold = np.percentile(cell_expr, 80)  # 80th percentile
        # 找出在这个细胞中表达量 > 阈值的基因
        top_genes_mask = cell_expr > threshold
        gene_count[top_genes_mask] += 1

    # 筛选在至少 min_cell_lines 个细胞中进入 top 80% 的基因
    filtered_mask = gene_count >= min_cell_lines
    filtered_genes = gene_names[filtered_mask].tolist()

    print(f"Filtered from {n_genes} to {len(filtered_genes)} genes.")
    return filtered_genes

if __name__ == "__main__":
    # Example usage
    tahoe_path = "gs://arc-ctc-tahoe100/2025-02-25/tutorial/plate3_2k-obs.h5ad"
    adata = load_tahoe_data(tahoe_path)
    filtered_genes = filter_proteins_by_expression(adata)
    print(f"Filtered genes: {len(filtered_genes)}")
