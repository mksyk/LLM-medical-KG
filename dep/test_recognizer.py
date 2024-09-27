import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json

# 定义科室列表
departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科', '中医科', '急诊科', '精神科', '营养科', '心理科', '男科', '其他科室']

def load_classifier(model_path, model_name="uer/sbert-base-chinese-nli"):
    """
    加载保存的分类器和基础模型
    """
    # 加载预训练的AutoModel和Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda()

    # 定义与训练时相同的分类器结构
    classifier = torch.nn.Linear(model.config.hidden_size, len(departments)).cuda()
    
    # 加载保存的权重
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()  # 切换到推理模式
    return tokenizer, model, classifier

def predict(query, tokenizer, model, classifier):
    """
    使用加载的分类器进行推理预测
    """
    # 对输入的query进行编码
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.cuda() for key, val in inputs.items()}

    # 获取句子嵌入
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token 的嵌入

    # 使用分类器进行预测
    logits = classifier(embeddings)
    probs = F.softmax(logits, dim=1)  # 计算每个类别的概率分布
    pred_label = torch.argmax(probs, dim=1)  # 获取概率最高的类别

    return departments[pred_label.item()], probs

# 加载分类器
model_path = "dep_model/classifier.pth"
tokenizer, model, classifier = load_classifier(model_path)

# 加载数据文件
data_file_path = "data/cmtMedQA_instruct.json"
with open(data_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 创建列表来存储每条数据的预测结果
results = []
# 初始化统计科室的计数
department_count = {dept: 0 for dept in departments}

# 处理每一条数据
for entry in data:
    query = entry["instruct"]
    predicted_label, probabilities = predict(query, tokenizer, model, classifier)

    # 更新对应科室的计数
    department_count[predicted_label] += 1

    # 将结果保存到字典
    result = {
        "query": query,
        "predicted_label": predicted_label,
        "probabilities": probabilities.detach().cpu().numpy().tolist()  # 将 tensor 转为列表
    }

    # 将处理后的数据加入结果列表
    results.append(result)

# 将结果和统计数据一起保存到 JSON 文件
output_data = {
    "results": results,
    "department_count": department_count
}

# 保存到新的 JSON 文件
output_file_path = "/predicted_results.json"
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# 输出每个科室的预测数量
print("各科室预测数量统计：")
for dept, count in department_count.items():
    print(f"{dept}: {count}")