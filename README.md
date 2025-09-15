## 🧪 项目工程 Prompt：构建细胞系特异性异质图用于药物-疾病关系预测

**项目名称**：CellLineKG — 基于细胞系特异性异质图的药物-疾病关系预测模型  
**核心目标**：预测“哪些药物能治疗某种癌症”，并输出潜在作用机制（靶点通路）。  
**基础架构**：完全仿照 **KGDRP** (DOI: 10.1002/advs.202412402)，数据源替换为 **Tahoe-100M** + **Enrichr** + **PINNACLE/ZINC**。  
**核心创新**：构建 **cell line-specific PPI**，并实现 **可配置的边构建策略**。

---

### 一、数据源与预处理

#### 1.1 节点类型 (Node Types)

*   `Drug`：来自 GDSC/PubChem，用 SMILES 表示，初始化为 1024-bit Morgan Fingerprint。
*   `Protein`：来自 Tahoe 数据。**过滤策略**：仅保留“在至少 N 个 cell line 中表达量位于 top 80%”的基因（N 为可配置参数，默认 N=3）。统一用 UniProt ID。
*   `CellLine`：直接使用 Tahoe 数据集中的 cell line（如“A549”, “MCF7”）。
*   `Disease`：仅选择癌症类型（如“Breast Cancer”, “Lung Adenocarcinoma”），与 Tahoe 中的 cell line 组织来源对应。
*   （可选）`Pathway` / `GO`：沿用 KGDRP 的 Reactome / UniProt-GO 数据，用于增强可解释性。

#### 1.2 边类型与构建方法（⭐重点：所有方法需支持 Config 配置）

| 边类型                      | 数据源               | 构建方法（Config Option）                                    | 边权重                                        |
| --------------------------- | -------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| **Drug – Protein**          | PINNACLE 或 ZINC     | 直接导入其预测的 DTI（二元或概率）                           | 概率值（若有）                                |
| **Protein – Protein (PPI)** | Tahoe 单细胞扰动数据 | **Option 1**: 基于**共表达**（Pearson/Spearman 相关性 > 阈值 `ppi_corr_threshold`，默认 0.7）<br>**Option 2**: 基于**扰动响应相似性**（不同扰动下基因表达变化向量的余弦相似度 > `ppi_perturb_threshold`，默认 0.6） | 相关性/相似度系数                             |
| **CellLine – Protein**      | Tahoe 基础表达数据   | 对每个 cell line，计算所有 gene 的表达量 Z-score。连接 Z-score > `expr_z_threshold`（默认 1.0）的 gene。 | Z-score 值（仅用于采样，不用于 GNN 消息传递） |
| **Disease – Protein**       | Enrichr              | 输入疾病名，获取其 Top K（默认 K=100）相关基因，建立二元边。 | 1 (或 Enrichr p-value)                        |
| **Disease – CellLine**      | 可配置双策略         | **Option A (基于表达)**：利用公共数据库（如 DepMap）找到该疾病 vs 正常组织的差异表达基因（DEGs），再连接到在 Tahoe 中高表达这些 DEGs 的 cell line。<br>**Option B (基于GWAS)**：利用 GWAS Catalog 找到疾病显著 SNP → 映射到基因 → 连接到在 Tahoe 中高表达这些基因的 cell line。 | DEG logFC / GWAS p-value                      |

> 📌 **工程要求**：在 `config.py` 中提供如下配置项：
>
> ```python
> # PPI构建策略
> PPI_BUILD_METHOD = "coexpression"  # or "perturbation"
> PPI_CORR_THRESHOLD = 0.7
> PPI_PERTURB_THRESHOLD = 0.6
> 
> # Disease-CellLine构建策略
> DISEASE_CELL_BUILD_METHOD = "de"  # or "gwas"
> 
> # Protein节点过滤
> MIN_CELL_LINES_FOR_TOP80 = 3
> 
> 
> ```

---

> 📌 **Tahoe 数据处理核心代码示例（已测试）**：
>
> ```python
> # 读取 Tahoe 数据
> infile = "gs://arc-ctc-tahoe100/2025-02-25/tutorial/plate3_2k-obs.h5ad"
> with fs.open(infile, 'rb') as f:
>  adata = sc.read_h5ad(f)
> 
> # 示例：计算两种药物处理下的基因表达差异 (log2FC)
> reference_drug = 'Everolimus'
> treatment_drug = 'Infigratinib'
> cell_line = 'NCI-H1792'
> 
> ref_cells = adata.obs[(adata.obs['drug'] == reference_drug) & (adata.obs['cell_name'] == cell_line)].index
> treat_cells = adata.obs[(adata.obs['drug'] == treatment_drug) & (adata.obs['cell_name'] == cell_line)].index
> 
> if len(ref_cells) > 0 and len(treat_cells) > 0:
>  mean_ref = np.mean(adata[ref_cells, :].X, axis=0).A1
>  mean_treat = np.mean(adata[treat_cells, :].X, axis=0).A1
>  log2fc = np.log2(mean_treat + 1.0) - np.log2(mean_ref + 1.0)
>  # 此 log2fc 向量可用于构建 "扰动响应相似性" PPI 或 Disease-CellLine 边
> ```

### 二、异质图构建 (Heterogeneous Graph Construction)

*   **框架**：使用 **DGL (Deep Graph Library)**，完全复刻 KGDRP 的图结构。
*   **图名称**：`CellLineBioHG`
*   **关键设计**：
    1.  `Drug` 和 `CellLine` 节点**不直接相连**。所有信息必须通过 `Protein` 节点传递，迫使模型学习生物学机制。
    2.  所有边都是**无向边**（除非后续实验表明有向边更优）。
    3.  `CellLine-Protein` 边的权重（Z-score）**仅用于生成训练样本**，**不参与 GNN 消息传递**（与 KGDRP 一致）。

---

### 三、模型架构与训练

*   **GNN 架构**：**完全沿用 KGDRP**。
    *   `Protein-Protein`, `Protein-Pathway`, `Drug-Protein` 边：使用 **GraphSAGE** 层。
    *   `CellLine-Protein` 边：使用 **GCN** 层。
    *   冷启动药物：使用线性变换 `h_drug = W_drug * x_drug` (x_drug = Morgan Fingerprint)。
*   **多任务学习**（辅助预测器）：
    1.  **RNA 表达预测器 (MLP)**：预测在某个 cell line 中，Gene A 的表达是否高于 Gene B。
    2.  **DTI 预测器 (MLP)**：预测 Drug X 是否与 Protein Y 相互作用。
    3.  **生物过程预测器 (MLP)**：预测 Protein Z 是否属于 Pathway P。
*   **主任务**：**Drug-Disease 关系预测**
    *   **输入**：Drug Embedding + Disease Embedding
    *   **模型**：一个简单的 **MLP 分类器**
    *   **输出**：概率值，表示“该药物能治疗该疾病”的可能性。
*   **损失函数**：沿用 KGDRP 的加权多任务损失：
    `Total Loss = w1 * Loss_RNA + w2 * Loss_DTI + w3 * Loss_Pathway + w4 * Loss_DrugDisease`

---

### 四、评估方案

*   **主任务指标**：
    *   **AUC-ROC**：衡量模型区分“有效药”和“无效药”的能力。
    *   **Precision@K / Recall@K** (K=10, 50, 100)：衡量在 Top K 预测中，有多少是真正的有效药。
*   **Baseline 模型**：
    1.  **原始 KGDRP** (在 GDSC 上训练)
    2.  **随机预测**
    3.  **MLP** (仅用药物指纹 + 疾病基因集平均embedding)
*   **消融实验**：
    1.  移除 `Disease-CellLine` 边。
    2.  移除 cell line-specific PPI，改用通用 PPI (如 STRING)。
    3.  关闭某个辅助任务（如 RNA 表达预测器）。

---

### 五、预期输出与交付物

1.  **代码库**：包含数据预处理、图构建、模型训练、评估的完整 pipeline。
2.  **配置文件** (`config.py`)：支持灵活切换 PPI 和 Disease-CellLine 的构建策略。
3.  **预训练模型**：在 Tahoe + Enrichr 数据上训练好的 CellLineKG 模型。
4.  **评估报告**：包含 AUC、P@K、R@K 等指标，以及与 Baseline 的对比。
5.  **机制分析示例**：对 Top 3 预测结果，输出类似 KGDRP Figure 5F 的“药物-靶点-细胞系-疾病”子网络，并标注关键枢纽蛋白（用中介中心性）。

---

### 六、下一步行动清单

1.  **数据下载**：获取 Tahoe-100M、Enrichr API、PINNACLE/ZINC DTI 数据。
2.  **ID Mapping**：将 Tahoe 的 gene symbol 映射到 UniProt ID。
3.  **实现 Config 模块**：先写好 `config.py`，定义所有可配置参数。
4.  **构建最小可行图 (MVP)**：先用一种 PPI 方法和一种 Disease-CellLine 方法，跑通整个流程。
5.  **迭代优化**：调整阈值、尝试不同策略，进行消融实验。

