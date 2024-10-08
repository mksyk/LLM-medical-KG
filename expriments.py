
import json
import os
from datetime import datetime
from multi_agent import run_medical_consultation, load_model_and_tokenizer
from cmner import check_bert_score
from torch.cuda import is_available
from bert_score import score
import random



def run_experiment(data_file_path, model_name, device, output_dir="results"):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成一个时间戳，用于创建唯一的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    
    # 创建实验目录
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # 加载数据集
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 随机抽取100条数据
    random_entries = random.sample(data, 100)

    results = []

    # 处理随机抽取的每个条目
    for entry in random_entries:
        query = entry["query"]
        reference_output = entry["output"]

        print(f"Processing query: {query}")
        
        # 获取系统输出和原模型输出
        system_output, ori_model_output = run_medical_consultation(query, model_name, device)
       
        # 计算BERT分数
        P_sys, R_sys, F1_sys = score([system_output], [reference_output], lang="zh", verbose=True)
        P_ori, R_ori, F1_ori = score([ori_model_output], [reference_output], lang="zh", verbose=True)

        bert_scores = {
            "precision": P_sys.mean().item() * 100,
            "recall": R_sys.mean().item() * 100,
            "f1": F1_sys.mean().item() * 100
        }

        ori_bert_scores = {
            "precision": P_ori.mean().item() * 100,
            "recall": R_ori.mean().item() * 100,
            "f1": F1_ori.mean().item() * 100
        }

        difference_f1 = (F1_sys.mean().item() - F1_ori.mean().item()) * 100

        # 存储当前条目的结果
        result = {
            "instruct": query,
            "reference_output": reference_output,
            "system_output": system_output,
            "ori_model_output": ori_model_output,
            "bert_scores": bert_scores,
            "ori_bert_scores": ori_bert_scores,
            "+/-": difference_f1
        }
        results.append(result)

    # 保存输出结果到 outputs.json
    outputs_file_path = os.path.join(experiment_dir, "outputs.json")
    with open(outputs_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Outputs saved to {outputs_file_path}")

    # 运行BERT分数检查并保存到 score.json
    scores = check_bert_score(outputs_file_path)
    score_file_path = os.path.join(experiment_dir, "scores.json")
    with open(score_file_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    print(f"Score saved to {score_file_path}")
# Example usage:
if __name__ == "__main__":
    llms_name = ['deepseek','llama', 'Qwen', 'baichuan', 'glm']
    model_name = 'deepseek'
    device = "cuda:7" if is_available() else "cpu"
    
    data_file_path = "datasets/CMtMedQA.json"
    run_experiment(data_file_path, model_name, device)
