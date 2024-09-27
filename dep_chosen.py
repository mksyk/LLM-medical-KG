import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dep.get_departments import get_dep
from dep.dep_recognizer_sft import MLPClassifier

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

    # 获取预测的类别和对应的概率
    predicted_label_idx = torch.argmax(probabilities, dim=1).item()
    predicted_label = departments[predicted_label_idx]
    predicted_probability = probabilities[0, predicted_label_idx].item()

    return predicted_label, predicted_probability

# 进行推断
query = "我有头痛，伴随着恶心想吐，请问该去哪个科室？"
predicted_label, predicted_probability = predict_department(query, tokenizer, transformer_model, classifier, device)

print(f"预测的科室是: {predicted_label}, 概率: {predicted_probability:.4f}")
