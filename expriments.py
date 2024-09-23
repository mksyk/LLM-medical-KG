import json
import os
from datetime import datetime
from multi_agent import run_medical_consultation, load_model_and_tokenizer
from torch.cuda import is_available
from bert_score import score

model_name = 'deepseek'
device = "cuda" if is_available() else "cpu"

if not os.path.exists("results"):
    os.makedirs("results")

data_file_path = "data/cmtMedQA_instruct.json"
with open(data_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def get_original_model_output(query,model_name,device):
    print('原始模型输出..')
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,max_new_tokens = 1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return response

results = []
for entry in data:
    query = entry["instruct"]
    reference_output = entry["output"]

    print(f"Processing query: {query}")
    
    system_output = run_medical_consultation(query, model_name, device)
    ori_model_output = get_original_model_output(query, model_name, device)

    P_sys, R_sys, F1_sys = score([system_output], [reference_output], lang="zh", verbose=True)
    P_ori, R_ori, F1_ori = score([ori_model_output], [reference_output], lang="zh", verbose=True)

    bert_scores = {
        "precision": P_sys.mean().item(),
        "recall": R_sys.mean().item(),
        "f1": F1_sys.mean().item()
    }

    ori_bert_scores = {
        "precision": P_ori.mean().item(),
        "recall": R_ori.mean().item(),
        "f1": F1_ori.mean().item()
    }

    difference_f1 = F1_sys.mean().item() - F1_ori.mean().item()

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file_path = f"results/{timestamp}.json"

with open(result_file_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {result_file_path}")
