
import json
import os
from datetime import datetime
from multi_agent import run_medical_consultation, load_model_and_tokenizer
from cmner import check_bert_score
from torch.cuda import is_available
from bert_score import score

def get_original_model_output(query, model_name, device):
    print('Getting original model output...')
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    if model_name=='baichuan':
        input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
        outputs = model.generate(input_ids,max_new_tokens=32)
    else: 
        inputs = tokenizer(query, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(query):].strip()
    return response

def run_experiment(data_file_path, model_name, device, output_dir="results"):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = []
    
    # Process each entry in the dataset (here limited to the first entry for demonstration)
    for entry in data:
        query = entry["instruct"]
        reference_output = entry["output"]

        print(f"Processing query: {query}")
        
        # Get the system output and original model output
        system_output = run_medical_consultation(query, model_name, device)
        ori_model_output = get_original_model_output(query, model_name, device)

        # Calculate BERT scores
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

        # Store results for the current entry
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

    # Save results to a JSON file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file_path = f"{output_dir}/{model_name}_{timestamp}.json"

    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {result_file_path}")

    # Run BERT score checking
    check_bert_score(result_file_path)

# Example usage:
if __name__ == "__main__":
    llms_name = ['llama', 'Qwen', 'deepseek', 'baichuan', 'glm']
    model_name = 'glm'
    device = "cuda:7" if is_available() else "cpu"
    
    data_file_path = "data/cmtMedQA_instruct.json"
    run_experiment(data_file_path, model_name, device)
