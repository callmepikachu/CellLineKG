"""
Training script for CellLineKG project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import CellLineKGModel
from train_sample_generator import TrainSampleGenerator
from config import *

def train_model(graph,
                drug_features,
                protein_features,
                cell_line_features=None,
                disease_features=None,
                device='cpu'):
    """
    Main training function.
    """
    # 初始化模型
    model = CellLineKGModel(
        protein_dim=protein_features.shape[1],
        drug_dim=drug_features.shape[1],
        cell_line_dim=cell_line_features.shape[1] if cell_line_features is not None else None,
        disease_dim=disease_features.shape[1] if disease_features is not None else None,
        hidden_dim=128,
        num_layers=2
    ).to(device)

    # 初始化样本生成器
    sampler = TrainSampleGenerator(graph, device=device)

    # 生成训练样本（简化：只生成一次）
    # 实际项目中应在每个epoch重新采样
    rna_samples = sampler.generate_rna_samples(graph.edges['cell_line_protein'].data['weight'])
    dti_samples = sampler.generate_dti_samples([('Everolimus', 'MTOR')])  # 示例，应替换为真实DTI
    known_pairs = [
        ('Everolimus', 'Breast Cancer'),
        ('Infigratinib', 'Lung Cancer'),
        ('Paclitaxel', 'Ovarian Cancer')
    ]
    drug_disease_samples = sampler.generate_drug_disease_samples(known_pairs, num_neg_per_pos=2)
    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    criterion = nn.BCELoss()

    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        node_embeddings, predictors = model(graph.to(device),
                                            drug_features.to(device),
                                            protein_features.to(device),
                                            cell_line_features.to(device) if cell_line_features is not None else None,
                                            disease_features.to(device) if disease_features is not None else None,
                                            task='all')

        total_loss = 0.0

        # --- RNA 任务 ---
        if len(rna_samples) > 0:
            cell_ids = rna_samples[:, 0]
            gene1_ids = rna_samples[:, 1]
            gene2_ids = rna_samples[:, 2]
            labels = rna_samples[:, 3].float()

            h_cell = node_embeddings['cell_line'][cell_ids]
            h_gene1 = node_embeddings['protein'][gene1_ids]
            h_gene2 = node_embeddings['protein'][gene2_ids]

            # 比较：gene1 是否 > gene2
            combined = torch.cat([h_gene1, h_gene2], dim=1)
            pred = predictors['rna'](combined).squeeze()
            loss_rna = criterion(pred, labels)
            total_loss += loss_rna

        # --- DTI 任务 ---
        if len(dti_samples) > 0:
            drug_ids = dti_samples[:, 0]
            prot_ids = dti_samples[:, 1]
            labels = dti_samples[:, 2].float()

            h_drug = node_embeddings['drug'][drug_ids]
            h_prot = node_embeddings['protein'][prot_ids]
            combined = torch.cat([h_drug, h_prot], dim=1)
            pred = predictors['dti'](combined).squeeze()
            loss_dti = criterion(pred, labels)
            total_loss += loss_dti

        # --- Drug-Disease 任务 ---
        if len(drug_disease_samples) > 0:
            drug_ids = drug_disease_samples[:, 0]
            dis_ids = drug_disease_samples[:, 1]
            labels = drug_disease_samples[:, 2].float()

            h_drug = node_embeddings['drug'][drug_ids]
            h_dis = node_embeddings['disease'][dis_ids]
            combined = torch.cat([h_drug, h_dis], dim=1)
            pred = predictors['drug_disease'](combined).squeeze()
            loss_dd = criterion(pred, labels)
            total_loss += loss_dd

        # 反向传播
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    print("✅ Training completed.")
    return model

if __name__ == "__main__":
    # 示例：你需要在这里加载真实数据并构建图
    print("Train script ready. Call train_model() with your data.")