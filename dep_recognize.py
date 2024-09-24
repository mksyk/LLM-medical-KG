'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-24 08:07:29
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-24 09:40:34
FilePath: /LLM-medical-KG/dep_recognize.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json
import os
import torch.nn as nn
import numpy as np




def extract_and_save_data(input_file, output_file):
    """
    从指定的 JSON 文件中提取 query, label1, label2 并保存为新的 JSON 文件。
    
    参数:
    - input_file (str): 输入 JSON 文件路径
    - output_file (str): 保存提取结果的输出 JSON 文件路径
    """
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 保存提取结果的列表
    result = []

    # 遍历每个条目，提取需要的数据
    for entry in data:
        # 判断 history 是否为空
        if entry["history"]:
            query = entry["history"][0][0]  # history 第一个列表中的第一个字符串
        else:
            query = entry["instruction"]  # 如果 history 为空，使用 instruction
        label1 = entry["cate1"]  # cate1 对应的 label1
        if label1 not in departments:
            label1 = "其他科室"  # 将 label1 改为 '其他科室'
        # 构造字典
        extracted_data = {
            "query": query,
            "label1": label1,
        }

        # 将结果添加到列表中
        result.append(extracted_data)

    # 将结果保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_file}")

class MedicalDepartmentFineTuner:
    def __init__(self, departments, model_name="bert-base-chinese", max_length=512, batch_size=8, num_epochs=3):
        """
        初始化微调类

        参数:
        - departments (list): 科室列表
        - model_name (str): 使用的预训练模型名称
        - max_length (int): 输入序列的最大长度
        - batch_size (int): 训练批次大小
        - num_epochs (int): 训练轮数
        """
        self.departments = departments
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(departments))
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
    
    def preprocess_data(self, data):
        """
        预处理数据，将 query 和 label 转化为模型输入

        参数:
        - data (list): 输入的数据，每一项是一个字典，包含 'query' 和 'label1'

        返回:
        - Dataset 对象，包含处理后的输入和标签
        """
        def preprocess_function(examples):
            # Tokenization
            inputs = self.tokenizer(examples['query'], padding="max_length", truncation=True, max_length=self.max_length)
            
            # One-hot 编码标签
            labels = [0] * len(self.departments)
            if examples['label1'] in self.departments:
                labels[self.departments.index(examples['label1'])] = 1
            
            inputs['labels'] = labels
            return inputs
        
        dataset = Dataset.from_list(data)
        dataset = dataset.map(preprocess_function)
        return dataset
    
    def train(self, train_data, eval_data):
        """
        训练模型

        参数:
        - train_data (list): 训练集数据
        - eval_data (list): 验证集数据
        """
        # 预处理训练和验证数据
        train_dataset = self.preprocess_data(train_data)
        eval_dataset = self.preprocess_data(eval_data)

        # 训练参数
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            fp16=True,  # 使用混合精度训练来提升性能
            logging_dir='./logs',  # 日志目录
            logging_steps=10
        )
        
        # 自定义损失函数为多标签二元交叉熵
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            sigmoid = nn.Sigmoid()
            probs = sigmoid(torch.Tensor(logits))
            predictions = (probs > 0.5).int()
            return {"accuracy": (predictions == labels).float().mean().item()}
        
        # 使用 Trainer 进行训练
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        # 开始训练
        trainer.train()

        # 保存模型
        self.model.save_pretrained("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")
        print("Model fine-tuned and saved.")

    def predict(self, query):
        """
        对新输入的 query 进行预测，并输出每个科室的概率分布

        参数:
        - query (str): 输入的 query 文本

        返回:
        - 科室列表中的概率分布
        """
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device)
        self.model.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            sigmoid = nn.Sigmoid()
            probs = sigmoid(logits).cpu().numpy()[0]  # 获取每个科室的概率分布
        
        # 返回科室及其对应的概率分布
        return list(zip(self.departments, probs))

departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科', '中医科', '急诊科', '精神科', '营养科', '心理科', '男科', '其他科室']

device = "cuda:7" if torch.cuda.is_available() else "cpu"

input_file = "data/cmtMedQA_chat.json"
output_file = "data/query_dep.json"
if not os.path.exists('data/query_dep.json'):
    extract_and_save_data(input_file, output_file)

with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 初始化微调类
fine_tuner = MedicalDepartmentFineTuner(departments)

train_size = int(0.8 * len(data)) 
train_data = data[:train_size]
eval_data = data[train_size:]  
fine_tuner.train(train_data, eval_data)

# 对新输入进行预测
query = "患者发热、咳嗽严重，伴随胸部疼痛。"
predictions = fine_tuner.predict(query)

# 输出预测结果
for department, score in predictions:
    print(f"{department}: {score:.4f}")