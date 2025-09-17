import os
import json

def collect_code_and_docs(target_dirs, output_file):
    with open(output_file, "w", encoding="utf-8") as out_f:
        for d in target_dirs:
            if not os.path.exists(d):
                print(f"目录 {d} 不存在，跳过。")
                continue
            for root, _, files in os.walk(d):
                for file in files:
                    # ✅ 改造点: 处理 .py 和 .ipynb 文件
                    if not (file.endswith(".py") or file.endswith(".ipynb")):
                        continue  # 跳过非Python和非Notebook文件
                    print(f"处理文件:{file}")

                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path)

                    try:
                        if file.endswith(".py"):
                            # 处理 Python 文件
                            print(f"处理文件:{file}")
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            out_f.write(f"## File: `{rel_path}`\n\n")
                            out_f.write("```python\n")
                            out_f.write(content)
                            out_f.write("\n```\n\n")

                        elif file.endswith(".ipynb"):
                            # 处理 Jupyter Notebook 文件
                            print(f"处理文件:{file}")

                            with open(file_path, "r", encoding="utf-8") as f:
                                notebook = json.load(f)

                            out_f.write(f"## File: `{rel_path}`\n\n")

                            # 遍历所有单元格
                            for i, cell in enumerate(notebook.get("cells", [])):
                                cell_type = cell.get("cell_type")
                                source = cell.get("source", [])

                                # 将列表形式的代码/文本拼接成字符串
                                if isinstance(source, list):
                                    source_text = "".join(source)
                                else:
                                    source_text = source

                                if cell_type == "code" and source_text.strip():
                                    # 代码单元格：用 ```python 包裹
                                    out_f.write(f"### Code Cell {i + 1}\n\n")
                                    out_f.write("```python\n")
                                    out_f.write(source_text)
                                    out_f.write("\n```\n\n")
                                elif cell_type == "markdown" and source_text.strip():
                                    # Markdown 单元格：直接输出，保留原有格式
                                    out_f.write(f"### Markdown Cell {i + 1}\n\n")
                                    out_f.write(source_text)
                                    out_f.write("\n\n")

                    except Exception as e:
                        print(f"跳过 {file_path} (无法读取或解析: {e})")
                        continue

                    out_f.write("---\n\n")  # 在每个文件后添加分隔线

if __name__ == "__main__":
    target_dirs = ["./utils", "."]  # 包含 .py 和 .ipynb 的目录
    output_file = "project_code.md"
    collect_code_and_docs(target_dirs, output_file)