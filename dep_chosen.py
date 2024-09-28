import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dep.get_departments import get_dep
from dep.dep_recognizer_sft import MLPClassifier
import json

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

def predict_department(query, tokenizer, transformer_model, classifier, device):
    # 对输入的query进行tokenizer编码
    inputs = tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 生成文本嵌入
    with torch.no_grad():
        embeddings = transformer_model(**inputs).last_hidden_state[:, 0, :]  # 使用 [CLS] token 的嵌入

    # 通过分类器得到类别概率分布
    with torch.no_grad():
        logits = classifier(embeddings)
        probabilities = F.softmax(logits, dim=1)  # 转化为概率分布

    # 获取整个概率分布并排序，取前3个最高的类别
    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

    # 构建返回结果
    top_labels = [departments[idx] for idx in top_indices[0].cpu().numpy()]
    top_probabilities = top_probs[0].cpu().numpy()

    result = {
        top_labels[i]: top_probabilities[i] * 100 for i in range(top_k)
    }

    return result

# 测试修改后的函数
data_file_path = 'data/huatuo_6000.json'
with open(data_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cnt = 0
for d in data:
    cnt+=1
    query = d['instruct']
    result = predict_department(query, tokenizer, transformer_model, classifier, device)
    
    output = {
        'query': query,
        'top_predictions': result
    }
    
    output_file_path = 'test_dep_cls.json'
    with open(output_file_path, "a", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"{cnt}: {result}")
