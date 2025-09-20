import pandas as pd
import json
import numpy as np
from typing import List, Dict


def get_all_drug_evidence(evidence_files: List[str], evidence_dir: str, all_disease: List[str],
                          chembl2db: Dict[str, str]):
    """
    Get all target-disease associations with clinically relevant evidence, i.e. mediated by approved drugs / clinical candidate >= II (must be 'Completed' if II)

    Args:
        evidence_files: List of evidence file names.
        evidence_dir: Directory path containing evidence files.
        all_disease: List of disease IDs to filter for.
        chembl2db: Dictionary mapping ChEMBL IDs to DrugBank IDs.

    Returns:
        pd.DataFrame: DataFrame containing filtered drug-target-disease evidence.
    """
    all_evidence = []
    total_records = 0  # 调试：计数器
    filtered_by_disease = 0
    filtered_by_phase = 0

    for file in evidence_files:
        evidence_file = evidence_dir + file
        # 读取 Parquet 文件
        df_evidence = pd.read_parquet(evidence_file)

        # 将 DataFrame 转换为字典列表，模拟原函数的 json.loads 行为
        evidence_list = df_evidence.to_dict(orient='records')

        for evidence in evidence_list:
            total_records += 1

            # --- 修复点 1: 使用正确的字段名 ---
            # 临床阶段
            clinical_phase = evidence.get('phase', 0)  # 修改: 'phase' 而非 'clinicalPhase'
            # 临床状态
            clinical_status = evidence.get('status', '')  # 修改: 'status' 而非 'clinicalStatus'

            # --- 修复点 2: 疾病ID匹配 ---
            # 尝试获取疾病ID，优先使用 diseaseFromSourceMappedId，其次 diseaseId
            disease_id = evidence.get('diseaseFromSourceMappedId') or evidence.get('diseaseId')
            # # 如果 disease_id 为空，或者不在 all_disease 列表中，则过滤
            # if not disease_id or disease_id not in all_disease:
            #     filtered_by_disease += 1
            #     continue

            # --- 修复点 3: 使用修复后的字段进行阶段过滤 ---
            if clinical_phase < 2:
                filtered_by_phase += 1
                continue
            if clinical_phase == 2 and clinical_status != 'Completed':
                filtered_by_phase += 1
                continue

            # 如果通过所有过滤，添加到结果
            db_id = chembl2db.get(evidence.get('drugId'), evidence.get('drugId')) if evidence.get('drugId') else evidence.get('drugId')
            evidence_entry = [
                disease_id,
                evidence.get('diseaseId', np.nan),
                evidence.get('targetId', np.nan),
                evidence.get('targetFromSourceId', np.nan),
                clinical_phase,  # 使用修正后的变量
                clinical_status,  # 使用修正后的变量
                db_id
            ]
            all_evidence.append(evidence_entry)

    print(f"\n--- 调试信息 ---")
    print(f"总读取记录数: {total_records}")
    print(f"因疾病ID被过滤: {filtered_by_disease}")
    print(f"因临床阶段被过滤: {filtered_by_phase}")
    print(f"最终保留记录数: {len(all_evidence)}")

    # 创建 DataFrame
    drug_evidence_data = pd.DataFrame(
        all_evidence,
        columns=[
            'diseaseFromSourceMappedId',
            'diseaseId',
            'targetId',
            'targetFromSourceId',
            'clinicalPhase',
            'clinicalStatus',
            'drugId'
        ]
    ).sort_values(by='targetId')

    # 数据验证
    if not drug_evidence_data.empty:
        # assert drug_evidence_data['diseaseFromSourceMappedId'].dropna().isin(
        #     all_disease).all(), "Found disease IDs not in all_disease list."
        assert drug_evidence_data['clinicalPhase'].dropna().isin([2, 3, 4]).all(), "Found invalid clinicalPhase values."

    # 保存为 CSV 文件
    output_csv_path = "drug2protein.csv"
    drug_evidence_data.to_csv(output_csv_path, index=False, sep='\t')
    print(f"Drug-protein evidence data saved to: {output_csv_path}")

    return drug_evidence_data


def load_chembl_to_drugbank_mapping(file_path: str) -> Dict[str, str]:
    """
    Load the ChEMBL to DrugBank ID mapping from a text file.
    Assumes the file has two columns separated by whitespace (e.g., tab or space).
    Format: ChEMBL_ID DrugBank_ID

    Args:
        file_path (str): Path to the mapping file (e.g., 'src1src2.txt').

    Returns:
        Dict[str, str]: A dictionary mapping ChEMBL IDs to DrugBank IDs.
    """
    chembl2db = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                chembl_id = parts[0]
                drugbank_id = parts[1]
                chembl2db[chembl_id] = drugbank_id
    return chembl2db


def load_disease_list_from_csv(file_path: str) -> List[str]:
    """
    Load a list of disease IDs from a CSV file.
    Assumes the CSV has a header and the disease IDs are in the first column.

    Args:
        file_path (str): Path to the CSV file (e.g., 'all_approved_oct2022.csv').

    Returns:
        List[str]: A list of disease IDs.
    """
    df = pd.read_csv(file_path)
    # 假设疾病ID在第一列，您可以根据实际情况修改列名，例如 df['diseaseId'].tolist()
    disease_list = df.iloc[:, 0].dropna().astype(str).tolist()
    return disease_list


def main():
    """
    Main function to orchestrate the extraction of drug-protein evidence data.
    """
    # >>>>>>>>>>>>>>>> 配置路径 <<<<<<<<<<<<<<<<
    EVIDENCE_DIR = ""  # 证据文件所在目录
    EVIDENCE_FILES = [
        "part-00000-f57fe3f5-70d3-4ebe-b9ca-b7dac348d291-c000.snappy.parquet",
        "part-00001-f57fe3f5-70d3-4ebe-b9ca-b7dac348d291-c000.snappy.parquet"
    ]
    CHEMBL2DB_PATH = "src1src2.txt"  # ChEMBL 到 DrugBank 的映射文件
    DISEASE_LIST_PATH = "all_approved_oct2022.csv"  # 包含疾病ID列表的文件

    # >>>>>>>>>>>>>>>> 加载数据 <<<<<<<<<<<<<<<<
    print("Loading ChEMBL to DrugBank mapping...")
    chembl2db_dict = load_chembl_to_drugbank_mapping(CHEMBL2DB_PATH)
    print(f"Loaded {len(chembl2db_dict)} ChEMBL to DrugBank mappings.")

    print("Loading disease ID list...")
    disease_id_list = load_disease_list_from_csv(DISEASE_LIST_PATH)
    print(f"Loaded {len(disease_id_list)} disease IDs.")

    # >>>>>>>>>>>>>>>> 调用核心函数 <<<<<<<<<<<<<<<<
    print("Processing drug evidence data...")
    drug_evidence_df = get_all_drug_evidence(
        evidence_files=EVIDENCE_FILES,
        evidence_dir=EVIDENCE_DIR,
        all_disease=disease_id_list,
        chembl2db=chembl2db_dict
    )

    # >>>>>>>>>>>>>>>> 输出最终结果 <<<<<<<<<<<<<<<<
    print(f"\nFinal drug-protein evidence data shape: {drug_evidence_df.shape}")
    print(drug_evidence_df.head())

    print("\nSummary:")
    print(f"Unique Drugs: {drug_evidence_df['drugId'].nunique()}")
    print(f"Unique Targets: {drug_evidence_df['targetId'].nunique()}")
    print(f"Unique Diseases: {drug_evidence_df['diseaseId'].nunique()}")

    print("\nDone!")


if __name__ == "__main__":
    main()