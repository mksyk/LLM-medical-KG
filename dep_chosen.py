import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dep.get_departments import get_dep
from dep.dep_recognizer_sft import MLPClassifier
import json
from cmner import predict_department
import random

departments,_ = get_dep()

# 模型配置
model_name = "uer/sbert-base-chinese-nli"  # 训练时使用的模型名称
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载分词器和预训练的Transformer模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name).to(device)

input_dim = 768  # 对应 SBERT/BERT 输出维度
hidden_dim = 512  # 你定义的隐藏层维度
output_dim = len(departments)  # 输出类别数（即科室的数量）

# 初始化模型
classifier = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)


# 加载保存的权重
classifier.load_state_dict(torch.load("dep_model/dep_classifier.pth"))

# 切换到评估模式
classifier.eval()
transformer_model.eval()

datasets_name = ['huatuo_26M','CMtMedQA','cMedQA-V2.0']

for dataset in datasets_name:
# 测试修改后的函数
    data_file_path = f"datasets/{dataset}.json"
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    random_entries = random.sample(data,100)
    for d in data[:20]:
        query = d['query']
        result = predict_department(query,departments, device)
        output = {
            'query': query,
            'top_predictions': result
        }
        
        output_file_path = 'test_dep_cls.json'
        with open(output_file_path, "a", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)