import json
from datasets import Dataset
from datasets import load_dataset

def save_dataset_to_json(dataset: Dataset, output_file: str):
    """
    将 Huggingface Dataset 对象保存为 JSON 文件。
    
    参数:
    - dataset: Huggingface Dataset 对象
    - output_file: 保存的 JSON 文件路径
    """
    # 将数据集转换为 Python 列表
    data = dataset['train'].to_dict()

    # 每个键是一个特征，每个特征包含所有实例的列表
    # 需要将其转为每个实例是一个字典的形式
    output_data = []
    num_examples = len(data[list(data.keys())[0]])  # 获取实例的数量

    for i in range(num_examples):
        entry = {key: data[key][i] for key in data}
        output_data.append(entry)

    # 保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"数据集已保存为 {output_file}")

# 示例使用
# 假设你已经加载了数据集
dataset = load_dataset("wangrongsheng/cMedQA-V2.0")
output_file = "data/cMedQA-V2.0.json"
save_dataset_to_json(dataset, output_file)
