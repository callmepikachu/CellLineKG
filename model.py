"""
Model definition for CellLineKG project.
Implements the GNN architecture based on KGDRP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, SAGEConv

class CellLineKGModel(nn.Module):
    """
    Main model for CellLineKG project.
    Implements heterogeneous GNN based on KGDRP.
    """

    def __init__(self,
                 protein_dim,
                 drug_dim=1024,  # 默认药物指纹维度
                 cell_line_dim=None,  # 细胞系特征维度（动态）
                 disease_dim=None,   # 疾病特征维度（动态）
                 hidden_dim=128,
                 num_layers=2):
        """
        Initialize the model.
        """
        super(CellLineKGModel, self).__init__()

        self.hidden_dim = hidden_dim

        # === 冷启动模块：药物指纹 → 药物表示 ===
        self.drug_linear_mapper = nn.Linear(drug_dim, hidden_dim)

        # === 蛋白、细胞系、疾病嵌入层 ===
        self.protein_embedding = nn.Linear(protein_dim, hidden_dim)

        # 细胞系和疾病如果传入维度，则创建嵌入层；否则用随机初始化（在forward中处理）
        if cell_line_dim is not None:
            self.cell_line_embedding = nn.Linear(cell_line_dim, hidden_dim)
        else:
            self.cell_line_embedding = None

        if disease_dim is not None:
            self.disease_embedding = nn.Linear(disease_dim, hidden_dim)
        else:
            self.disease_embedding = None

        # === GNN Layers ===
        self.sage_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))

        # GCN layer for cell_line-protein edges
        self.gcn_layer = GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True)

        # === Auxiliary Predictors (MLP) ===
        self.rna_predictor = self._build_mlp(hidden_dim * 2)
        self.dti_predictor = self._build_mlp(hidden_dim * 2)
        self.pathway_predictor = self._build_mlp(hidden_dim * 2)

        # === Main Task: Drug-Disease Predictor ===
        self.drug_disease_predictor = self._build_mlp(hidden_dim * 2)

    def _build_mlp(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, graph,
                drug_features,
                protein_features,
                cell_line_features=None,
                disease_features=None,
                task='all'):
        """
        Forward pass with heterogeneous message passing.

        Args:
            graph: DGL heterogeneous graph.
            drug_features: Tensor of shape (n_drugs, drug_dim) — Morgan Fingerprint.
            protein_features: Tensor of shape (n_proteins, protein_dim) — 可以是随机初始化或预训练嵌入。
            cell_line_features: Optional. Tensor of shape (n_cell_lines, cell_line_dim).
            disease_features: Optional. Tensor of shape (n_diseases, disease_dim).
            task: 'all', 'drug_disease', 'rna', 'dti', 'pathway'
        """
        # === Step 1: Initialize Node Embeddings ===
        # Drug: 通过线性映射（支持冷启动）
        h_drug = self.drug_linear_mapper(drug_features)  # (n_drugs, hidden_dim)

        # Protein
        h_protein = self.protein_embedding(protein_features)  # (n_proteins, hidden_dim)

        # Cell Line
        if cell_line_features is not None and self.cell_line_embedding is not None:
            h_cell_line = self.cell_line_embedding(cell_line_features)
        else:
            # 如果没有传入特征，使用可学习的嵌入（随机初始化）
            n_cell_lines = graph.num_nodes('cell_line')
            if not hasattr(self, 'cell_line_embed'):
                self.cell_line_embed = nn.Parameter(torch.randn(n_cell_lines, self.hidden_dim))
            h_cell_line = self.cell_line_embed

        # Disease
        if disease_features is not None and self.disease_embedding is not None:
            h_disease = self.disease_embedding(disease_features)
        else:
            n_diseases = graph.num_nodes('disease')
            if not hasattr(self, 'disease_embed'):
                self.disease_embed = nn.Parameter(torch.randn(n_diseases, self.hidden_dim))
            h_disease = self.disease_embed

        # 保存初始蛋白特征用于残差连接
        h_protein_init = h_protein.clone()

        # === Step 2: Message Passing ===
        # 保存初始蛋白特征用于最终残差连接
        h_protein_init = h_protein.clone()

        # Protein-Protein edges (GraphSAGE)
        for sage_layer in self.sage_layers:
            h_protein_new = sage_layer(graph, h_protein, etype='protein_protein')
            h_protein_new = F.relu(h_protein_new)
            h_protein_new = F.dropout(h_protein_new, p=0.3, training=self.training)
            h_protein = h_protein + h_protein_new  # 残差连接

        # CellLine-Protein edges (GCN)
        if graph.num_edges('cell_line_protein') > 0:
            clp_subgraph = dgl.edge_type_subgraph(graph, [('cell_line', 'cell_line_protein', 'protein')])
            h_protein_clp = self.gcn_layer(clp_subgraph, (h_cell_line, h_protein))
            h_protein_clp = F.relu(h_protein_clp)
            h_protein_clp = F.dropout(h_protein_clp, p=0.3, training=self.training)
            h_protein = h_protein + h_protein_clp  # 残差连接

        h_protein = h_protein + h_protein_init

        # === Step 3: Prepare Output Embeddings ===
        node_embeddings = {
            'drug': h_drug,
            'protein': h_protein,
            'cell_line': h_cell_line,
            'disease': h_disease
        }

        # 如果只要求返回嵌入（如用于采样），直接返回
        if task == 'embeddings':
            return node_embeddings

        # === Step 4: 执行预测任务 ===
        outputs = {}

        if task in ['all', 'rna']:
            # RNA表达比较任务：需要外部采样 (cell, gene1, gene2)
            outputs['rna'] = self.rna_predictor

        if task in ['all', 'dti']:
            # DTI预测：需要外部采样 (drug, protein)
            outputs['dti'] = self.dti_predictor

        if task in ['all', 'pathway']:
            # Pathway预测：需要外部采样 (protein, pathway) — 注意：当前图无pathway节点
            outputs['pathway'] = self.pathway_predictor

        if task in ['all', 'drug_disease']:
            # 主任务：Drug-Disease 预测
            outputs['drug_disease'] = self.drug_disease_predictor

        return node_embeddings, outputs

    def predict_drug_disease(self, drug_emb, disease_emb):
        """
        预测药物-疾病对的得分。
        Args:
            drug_emb: 单个药物嵌入 (hidden_dim,)
            disease_emb: 单个疾病嵌入 (hidden_dim,)
        Returns:
            score: 标量，0~1之间
        """
        combined = torch.cat([drug_emb, disease_emb], dim=-1).unsqueeze(0)  # (1, 2*hidden_dim)
        score = self.drug_disease_predictor(combined).squeeze()  # scalar
        return score