
import json
import os
from datetime import datetime
from multi_agent import run_medical_consultation, load_model_and_tokenizer
from cmner import check_score,save_dict_with_spacing
from torch.cuda import is_available
from bert_score import score
import random
import os
import json
import random
from datetime import datetime
import sacrebleu
from rouge_score import rouge_scorer
import re


def run_experiment(data_file_path, model_name, device, dataset, output_dir="results"):
    # 加载数据集
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cnt_data = 50
    random_entries = random.sample(data,cnt_data)

    results = []
    
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    right_disease = 0 
    # 处理随机抽取的每个条目    
    for entry in random_entries:
        query = entry["query"]
        reference_output = entry["output"]

        print(f"Processing query: {query}")
        # 获取系统输出和原模型输出
        system_output, ori_model_output,rd = run_medical_consultation(query,model,tokenizer, device)
        if rd:
            right_disease += 1

        # 计算BERT分数
        P_sys, R_sys, F1_sys = score([system_output], [reference_output], lang="zh", verbose=True)
        P_ori, R_ori, F1_ori = score([ori_model_output], [reference_output], lang="zh", verbose=True)


        # BERT score results
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
            "+/-_bert_f1": difference_f1,
        } 
        

        # 调用存储函数
        save_dict_with_spacing(result, 'test_outputs.md')
        results.append(result)
     # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    # 生成一个时间戳，用于创建唯一的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{model_name}_{dataset}_{timestamp}")
    
    # 创建实验目录
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    # 保存输出结果到 outputs.json
    outputs_file_path = os.path.join(experiment_dir, "outputs.json")
    with open(outputs_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Outputs saved to {outputs_file_path}")

    # 运行BERT分数检查并保存到 score.json
    scores = check_score(outputs_file_path)
    score_file_path = os.path.join(experiment_dir, "scores.json")
    with open(score_file_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    print(f"Score saved to {score_file_path}")
    print(f'right_disease = {right_disease}')

if __name__ == "__main__":
    llms_name = ['deepseek','llama', 'Qwen', 'baichuan', 'glm','Qwen2']
    model_name = 'deepseek'
    device = "cuda:6" if is_available() else "cpu"
    datasets_name = ['huatuo_26M','CMtMedQA','cMedQA-V2.0']
    # dataset = 'huatuo_26M'
    dataset = 'huatuo_26M'
    # for dataset in datasets_name:
    data_file_path = f"datasets/{dataset}.json"
    run_experiment(data_file_path, model_name, device,dataset)
