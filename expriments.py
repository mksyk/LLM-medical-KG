
import json
import os
from datetime import datetime
from multi_agent import run_medical_consultation, load_model_and_tokenizer
from cmner import check_score,save_to_md
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
    
    random_entries = random.sample(data,20)

    results = []
    
    # 初始化ROUGE评分器
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    # 处理随机抽取的每个条目    
    for entry in random_entries:
        query = entry["query"]
        reference_output = entry["output"]

        print(f"Processing query: {query}")
        
        # 获取系统输出和原模型输出
        system_output, ori_model_output = run_medical_consultation(query,model,tokenizer, device)

        # 计算BERT分数
        P_sys, R_sys, F1_sys = score([system_output], [reference_output], lang="zh", verbose=True)
        P_ori, R_ori, F1_ori = score([ori_model_output], [reference_output], lang="zh", verbose=True)

        # 计算 BLEU 分数
        bleu_sys = sacrebleu.corpus_bleu([system_output], [[reference_output]]).score
        bleu_ori = sacrebleu.corpus_bleu([ori_model_output], [[reference_output]]).score

        # 计算ROUGE分数 (直接基于标点符号进行简单分句)
        def simple_sentence_split(text):
            # 基于中文句号、问号、叹号进行分割
            return [sentence for sentence in re.split(r'(?<=[。？！])', text) if sentence.strip()]

        system_sentences = simple_sentence_split(system_output)
        reference_sentences = simple_sentence_split(reference_output)
        ori_model_sentences = simple_sentence_split(ori_model_output)

        # 初始化ROUGE累积分数
        rouge1_sys, rouge2_sys, rougeL_sys = 0, 0, 0
        rouge1_ori, rouge2_ori, rougeL_ori = 0, 0, 0

        # 逐句计算ROUGE分数
        for ref_sentence, sys_sentence, ori_sentence in zip(reference_sentences, system_sentences, ori_model_sentences):
            rouge_sys_scores = rouge_scorer_instance.score(ref_sentence, sys_sentence)
            rouge_ori_scores = rouge_scorer_instance.score(ref_sentence, ori_sentence)

            rouge1_sys += rouge_sys_scores['rouge1'].fmeasure
            rouge2_sys += rouge_sys_scores['rouge2'].fmeasure
            rougeL_sys += rouge_sys_scores['rougeL'].fmeasure

            rouge1_ori += rouge_ori_scores['rouge1'].fmeasure
            rouge2_ori += rouge_ori_scores['rouge2'].fmeasure
            rougeL_ori += rouge_ori_scores['rougeL'].fmeasure

        # 计算句子平均ROUGE分数
        num_sentences = len(reference_sentences)
        rouge1_sys /= num_sentences
        rouge2_sys /= num_sentences
        rougeL_sys /= num_sentences

        rouge1_ori /= num_sentences
        rouge2_ori /= num_sentences
        rougeL_ori /= num_sentences

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

        # BLEU score difference
        difference_bleu = (bleu_sys - bleu_ori)

        # ROUGE score difference (rouge1, rouge2, rougeL)
        difference_rouge1 = (rouge1_sys - rouge1_ori) * 100
        difference_rouge2 = (rouge2_sys - rouge2_ori) * 100
        difference_rougeL = (rougeL_sys - rougeL_ori) * 100

        # 存储当前条目的结果
        result = {
            "instruct": query,
            "reference_output": reference_output,
            "system_output": system_output,
            "ori_model_output": ori_model_output,
            "bert_scores": bert_scores,
            "ori_bert_scores": ori_bert_scores,
            "+/-_bert_f1": difference_f1,
            "bleu_sys": bleu_sys,
            "bleu_ori": bleu_ori,
            "+/-_bleu": difference_bleu,
            "rouge_sys": {
                "rouge1": rouge1_sys * 100,
                "rouge2": rouge2_sys * 100,
                "rougeL": rougeL_sys * 100
            },
            "rouge_ori": {
                "rouge1": rouge1_ori * 100,
                "rouge2": rouge2_ori * 100,
                "rougeL": rougeL_ori * 100
            },
            "+/-_rouge1": difference_rouge1,
            "+/-_rouge2": difference_rouge2,
            "+/-_rougeL": difference_rougeL
        }
        save_to_md('test_ouputs.md',f'{result}')
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

if __name__ == "__main__":
    llms_name = ['deepseek','llama', 'Qwen', 'baichuan', 'glm','Qwen2']
    model_name = 'deepseek'
    device = "cuda:7" if is_available() else "cpu"
    datasets_name = ['huatuo_26M','CMtMedQA','cMedQA-V2.0']
    # dataset = 'huatuo_26M'
    dataset = 'cMedQA-V2.0'
    data_file_path = f"datasets/{dataset}.json"
    run_experiment(data_file_path, model_name, device,dataset)
