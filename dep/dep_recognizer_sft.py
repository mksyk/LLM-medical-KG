import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset
import json
import os
from cre_query_dep import extract_and_save_data
from get_departments import get_dep

departments,_ = get_dep()

def load_and_process_data(output_file):
    """
    加载处理好的 JSON 数据，并将其转换为 Huggingface Dataset 格式。
    """
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [entry['query'] for entry in data]
    labels = [departments.index(entry['label']) for entry in data]
    
    dataset = Dataset.from_dict({
        'query': queries,
        'label': labels
    })
    return dataset

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        # 两层 MLP
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)            # 第一层全连接
        x = self.relu(x)           # 激活函数
        x = self.dropout(x)        # Dropout（防止过拟合）
        x = self.fc2(x)            # 第二层全连接，输出
        return x

def train_model(dataset, model_name="uer/sbert-base-chinese-nli", output_dir="dep_model", num_labels=len(departments), num_epochs=2, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    使用 AutoModel 和 AutoTokenizer 进行科室分类，并划分训练集和验证集。
    """
    # 划分训练集和验证集 (80%训练集，20%验证集)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    def collate_fn(batch):
        queries = [example['query'] for example in batch]
        labels = torch.tensor([example['label'] for example in batch], dtype=torch.long)

        # 编码输入，设置 max_length 为模型的最大输入长度，防止超长文本导致错误
        inputs = tokenizer(queries, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 生成句子嵌入
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # 使用 [CLS] token 的嵌入

        return embeddings, labels

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 定义两层 MLP 分类器
    hidden_dim = 512
    classifier = MLPClassifier(input_dim=model.config.hidden_size, hidden_dim=hidden_dim, output_dim=num_labels).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    # 训练和验证循环
    classifier.train()
    model.train()  # 切换到训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch_idx, (batch_embeddings, batch_labels) in enumerate(train_loader):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            outputs = classifier(batch_embeddings)
            loss = F.cross_entropy(outputs, batch_labels)
            total_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {correct_predictions / total_predictions:.4f}")


        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}, Training Accuracy: {train_accuracy}")

        # 验证模型并输出科室的概率分布
        classifier.eval()
        model.eval()  # 切换到验证模式
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for val_embeddings, val_labels in val_loader:
                val_embeddings = val_embeddings.to(device)
                val_labels = val_labels.to(device)

                val_outputs = classifier(val_embeddings)
                val_loss = F.cross_entropy(val_outputs, val_labels)
                total_val_loss += val_loss.item()

                # 计算概率分布
                probabilities = F.softmax(val_outputs, dim=1)
                print(f"Probabilities:\n{probabilities}")

                # 计算验证集准确率
                _, predicted = torch.max(val_outputs, 1)
                correct_predictions += (predicted == val_labels).sum().item()
                total_predictions += val_labels.size(0)

        val_accuracy = correct_predictions / total_predictions
        print(f"Validation Loss: {total_val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}")

        classifier.train()
        model.train()  # 切换回训练模式

    # 保存模型
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_save_path = os.path.join(output_dir, "mlp_classifier.pth")
    torch.save(classifier.state_dict(), model_save_path)
    print(f"MLP Classifier model saved to {model_save_path}")

# 文件路径
input_file = "data/CMtMedQA_mapped.json"
output_file = "data/query_dep.json"

# 如果尚未处理数据，则提取数据
if not os.path.exists(output_file):
    extract_and_save_data(input_file, output_file)

# 加载并处理数据
dataset = load_and_process_data(output_file)

# 模型训练，指定设备为 'cpu' 或 'cuda' 
train_model(dataset, model_name='uer/sbert-base-chinese-nli', device="cuda" if torch.cuda.is_available() else "cpu")
