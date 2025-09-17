# File: train_sample_generator.py
"""
Training sample generator for CellLineKG.
Converts regression tasks into comparison tasks, as in KGDRP.
"""

import torch
import random
from collections import defaultdict

class TrainSampleGenerator:
    def __init__(self, graph, device='cpu'):
        self.graph = graph
        self.device = device
        self._build_index_maps()

    def _build_index_maps(self):
        """构建节点名到ID的映射"""
        self.protein_name_to_id = {name: i for i, name in enumerate(self.graph.nodes['protein'].data['_ID'].tolist())}
        self.drug_name_to_id = {name: i for i, name in enumerate(self.graph.nodes['drug'].data['_ID'].tolist())}
        self.cell_line_name_to_id = {name: i for i, name in enumerate(self.graph.nodes['cell_line'].data['_ID'].tolist())}
        self.disease_name_to_id = {name: i for i, name in enumerate(self.graph.nodes['disease'].data['_ID'].tolist())}

    def generate_rna_samples(self, cell_line_protein_weights, num_samples=10000):
        """
        生成RNA表达比较样本: (cell_line, gene_high, gene_low)
        使用 edge weight (表达量Z-score) 作为采样依据
        """
        samples = []
        edges = self.graph.edges(etype='cell_line_protein', form='all')
        src, dst, eid = edges
        weights = self.graph.edges['cell_line_protein'].data['weight']

        # 按cell line分组
        cell_gene_pairs = defaultdict(list)
        for i in range(len(src)):
            cell_id = src[i].item()
            gene_id = dst[i].item()
            weight = weights[i].item()
            cell_gene_pairs[cell_id].append((gene_id, weight))

        # 对每个cell line，生成gene对
        for cell_id, gene_list in cell_gene_pairs.items():
            if len(gene_list) < 2:
                continue
            for _ in range(num_samples // len(cell_gene_pairs)):
                g1, w1 = random.choice(gene_list)
                g2, w2 = random.choice(gene_list)
                if w1 > w2:
                    samples.append((cell_id, g1, g2, 1))  # g1 > g2
                elif w1 < w2:
                    samples.append((cell_id, g1, g2, 0))  # g1 < g2
                # 相等则跳过

        return torch.tensor(samples, device=self.device)

    def generate_dti_samples(self, positive_edges, num_neg_per_pos=1):
        """
        生成DTI样本 (drug, protein, label)
        使用负采样
        """
        samples = []
        # 正样本
        for drug, prot in positive_edges:
            d_id = self.drug_name_to_id[drug]
            p_id = self.protein_name_to_id[prot]
            samples.append((d_id, p_id, 1))

        # 负样本：随机采样
        all_proteins = list(self.protein_name_to_id.values())
        for drug, prot in positive_edges:
            d_id = self.drug_name_to_id[drug]
            for _ in range(num_neg_per_pos):
                p_id = random.choice(all_proteins)
                samples.append((d_id, p_id, 0))

        return torch.tensor(samples, device=self.device)

    def generate_drug_disease_samples(self, known_drug_disease_pairs, num_neg_per_pos=1):
        """
        生成药物-疾病样本 (drug, disease, label)
        使用真实药-病对作为正样本，负采样构建负样本

        Args:
            known_drug_disease_pairs (list of tuples): 真实药-病对，如 [('Aspirin', 'Pain'), ...]
            num_neg_per_pos (int): 每个正样本对应的负样本数

        Returns:
            torch.Tensor: shape (N, 3) — [drug_id, disease_id, label]
        """
        samples = []

        # 正样本
        for drug, disease in known_drug_disease_pairs:
            if drug not in self.drug_name_to_id or disease not in self.disease_name_to_id:
                continue
            d_id = self.drug_name_to_id[drug]
            dis_id = self.disease_name_to_id[disease]
            samples.append((d_id, dis_id, 1))

        # 负样本：对每个正样本，随机替换 drug 或 disease
        all_drugs = list(self.drug_name_to_id.keys())
        all_diseases = list(self.disease_name_to_id.keys())

        for drug, disease in known_drug_disease_pairs:
            if drug not in self.drug_name_to_id or disease not in self.disease_name_to_id:
                continue
            dis_id = self.disease_name_to_id[disease]
            for _ in range(num_neg_per_pos):
                # 随机选一个不在正样本中的药物
                neg_drug = random.choice(all_drugs)
                while (neg_drug, disease) in known_drug_disease_pairs:
                    neg_drug = random.choice(all_drugs)
                d_id = self.drug_name_to_id[neg_drug]
                samples.append((d_id, dis_id, 0))

            for _ in range(num_neg_per_pos):
                # 随机选一个不在正样本中的疾病
                neg_disease = random.choice(all_diseases)
                while (drug, neg_disease) in known_drug_disease_pairs:
                    neg_disease = random.choice(all_diseases)
                dis_id_neg = self.disease_name_to_id[neg_disease]
                d_id = self.drug_name_to_id[drug]
                samples.append((d_id, dis_id_neg, 0))

        if len(samples) == 0:
            return torch.empty((0, 3), dtype=torch.long, device=self.device)

        return torch.tensor(samples, dtype=torch.long, device=self.device)