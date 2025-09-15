# Configuration file for CellLineKG project

# PPI construction strategy
PPI_BUILD_METHOD = "coexpression"  # Options: "coexpression" or "perturbation"
PPI_CORR_THRESHOLD = 0.7
PPI_PERTURB_THRESHOLD = 0.6

# Disease-CellLine construction strategy
DISEASE_CELL_BUILD_METHOD = "de"  # Options: "de" (differential expression) or "gwas"

# Protein node filtering
MIN_CELL_LINES_FOR_TOP80 = 3

# CellLine-Protein edge threshold
EXPR_Z_THRESHOLD = 1.0

# Disease-Protein edge top K genes
DISEASE_PROTEIN_TOP_K = 100