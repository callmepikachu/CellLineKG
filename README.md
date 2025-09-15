# CellLineKG - Cell Line-specific Heterogeneous Graph for Drug-Disease Relationship Prediction

## Project Overview

**Project Name**: CellLineKG — Cell Line-specific Heterogeneous Graph for Drug-Disease Relationship Prediction  
**Core Objective**: Predict "which drugs can treat which cancers" and output potential mechanisms of action (target pathways).  
**Base Architecture**: Fully replicates **KGDRP** (DOI: 10.1002/advs.202412402), with data sources replaced by **Tahoe-100M** + **Enrichr** + **PINNACLE/ZINC**.  
**Key Innovation**: Construction of **cell line-specific PPI** and implementation of **configurable edge construction strategies**.

## Data Sources and Preprocessing

### Node Types

*   `Drug`: From GDSC/PubChem, represented by SMILES, initialized as 1024-bit Morgan Fingerprint.
*   `Protein`: From Tahoe data. **Filtering Strategy**: Only retain genes "expressed in the top 80% in at least N cell lines" (N is a configurable parameter, default N=3). Standardized to UniProt ID.
*   `CellLine`: Directly use cell lines from the Tahoe dataset (e.g., "A549", "MCF7").
*   `Disease`: Only select cancer types (e.g., "Breast Cancer", "Lung Adenocarcinoma"), corresponding to the tissue origin of cell lines in Tahoe.
*   (Optional) `Pathway` / `GO`: Replicate Reactome / UniProt-GO data from KGDRP to enhance interpretability.

### Edge Types and Construction Methods (⭐Key: All methods support Config configuration)

| Edge Type                   | Data Source          | Construction Method (Config Option)                          | Edge Weight                                   |
| --------------------------- | -------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| **Drug – Protein**          | PINNACLE or ZINC     | Directly import predicted DTI (binary or probability)        | Probability value (if available)              |
| **Protein – Protein (PPI)** | Tahoe single-cell perturbation data | **Option 1**: Based on **co-expression** (Pearson/Spearman correlation > threshold `ppi_corr_threshold`, default 0.7)<br>**Option 2**: Based on **perturbation response similarity** (cosine similarity of gene expression change vectors under different perturbations > `ppi_perturb_threshold`, default 0.6) | Correlation/similarity coefficient            |
| **CellLine – Protein**      | Tahoe baseline expression data | For each cell line, calculate Z-score of all gene expression levels. Connect genes with Z-score > `expr_z_threshold` (default 1.0). | Z-score value (only for sampling, not for GNN message passing) |
| **Disease – Protein**       | Enrichr              | Input disease name, obtain Top K (default K=100) related genes, establish binary edges. | 1 (or Enrichr p-value)                        |
| **Disease – CellLine**      | Configurable dual strategies | **Option A (Expression-based)**: Use public databases (e.g., DepMap) to find differentially expressed genes (DEGs) between the disease and normal tissues, then connect to cell lines with high expression of these DEGs in Tahoe.<br>**Option B (GWAS-based)**: Use GWAS Catalog to find disease-significant SNPs → map to genes → connect to cell lines with high expression of these genes in Tahoe. | DEG logFC / GWAS p-value                      |

## Heterogeneous Graph Construction

*   **Framework**: Use **DGL (Deep Graph Library)**, fully replicating KGDRP's graph structure.
*   **Graph Name**: `CellLineBioHG`
*   **Key Design**:
    1.  `Drug` and `CellLine` nodes **do not directly connect**. All information must be transmitted through `Protein` nodes, forcing the model to learn biological mechanisms.
    2.  All edges are **undirected edges** (unless subsequent experiments show directed edges are superior).
    3.  The weights of `CellLine-Protein` edges (Z-score) **are only used to generate training samples**, **not participating in GNN message passing** (consistent with KGDRP).

## Model Architecture and Training

*   **GNN Architecture**: **Fully replicates KGDRP**.
    *   `Protein-Protein`, `Protein-Pathway`, `Drug-Protein` edges: Use **GraphSAGE** layers.
    *   `CellLine-Protein` edges: Use **GCN** layers.
    *   Cold-start drugs: Use linear transformation `h_drug = W_drug * x_drug` (x_drug = Morgan Fingerprint).
*   **Multi-task Learning** (Auxiliary Predictors):
    1.  **RNA Expression Predictor (MLP)**: Predict whether Gene A's expression is higher than Gene B's in a cell line.
    2.  **DTI Predictor (MLP)**: Predict whether Drug X interacts with Protein Y.
    3.  **Biological Process Predictor (MLP)**: Predict whether Protein Z belongs to Pathway P.
*   **Main Task**: **Drug-Disease Relationship Prediction**
    *   **Input**: Drug Embedding + Disease Embedding
    *   **Model**: A simple **MLP classifier**
    *   **Output**: Probability value, indicating the likelihood that "the drug can treat the disease".
*   **Loss Function**: Replicates KGDRP's weighted multi-task loss:
    `Total Loss = w1 * Loss_RNA + w2 * Loss_DTI + w3 * Loss_Pathway + w4 * Loss_DrugDisease`

## Evaluation Scheme

*   **Main Task Metrics**:
    *   **AUC-ROC**: Measure the model's ability to distinguish "effective drugs" from "ineffective drugs".
    *   **Precision@K / Recall@K** (K=10, 50, 100): Measure how many of the Top K predictions are truly effective drugs.
*   **Baseline Models**:
    1.  **Original KGDRP** (trained on GDSC)
    2.  **Random Prediction**
    3.  **MLP** (only using drug fingerprints + average embedding of disease gene sets)
*   **Ablation Experiments**:
    1.  Remove `Disease-CellLine` edges.
    2.  Remove cell line-specific PPI, replace with generic PPI (e.g., STRING).
    3.  Disable a specific auxiliary task (e.g., RNA expression predictor).

## Expected Outputs and Deliverables

1.  **Codebase**: Complete pipeline including data preprocessing, graph construction, model training, and evaluation.
2.  **Configuration File** (`config.py`): Supports flexible switching of PPI and Disease-CellLine construction strategies.
3.  **Pre-trained Model**: CellLineKG model trained on Tahoe + Enrichr data.
4.  **Evaluation Report**: Includes AUC, P@K, R@K metrics and comparison with Baselines.
5.  **Mechanism Analysis Example**: For the Top 3 predictions, output a "drug-target-cell line-disease" subnetwork similar to KGDRP Figure 5F, and annotate key hub proteins (using betweenness centrality).

## Next Action Items

1.  **Data Download**: Obtain Tahoe-100M, Enrichr API, PINNACLE/ZINC DTI data.
2.  **ID Mapping**: Map Tahoe's gene symbols to UniProt IDs.
3.  **Implement Config Module**: Write `config.py` first, defining all configurable parameters.
4.  **Build Minimum Viable Graph (MVP)**: Use one PPI method and one Disease-CellLine method to run the entire pipeline.
5.  **Iterative Optimization**: Adjust thresholds, try different strategies, conduct ablation experiments.