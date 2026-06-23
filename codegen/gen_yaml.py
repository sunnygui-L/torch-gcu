# 安装依赖库（如果有）
try:
    import yaml  # 检查 yaml 库是否已经安装
except ImportError:
    import os
    os.system("pip install PyYAML")
    print("PyYAML library installed.")
import yaml
import os


def process_yaml(input_file):
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)
        original_data = data.copy()

    if not data:
        print("Error: Empty or invalid YAML file.")
        return

    # 处理带有 "-" 标记的部分并按字母排序
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = sorted(value, key=lambda x:x["name"])

    # 写回原文件
    if data == original_data:
        print("No changes made to YAML file.")
        return
    with open(input_file, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    print(f"Processed YAML saved to {input_file}")


if __name__ == "__main__":
    # get file path
    current_path = os.path.dirname(os.path.abspath(__file__))
    input_yaml_file = os.path.join(current_path, "gcu_native_functions.yaml")

    # 处理 YAML 文件
    process_yaml(input_yaml_file)
