'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-19 03:24:50
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-23 09:56:16
FilePath: /LLM-medical-KG/department_KB_generate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig
import time
import torch
from cmner import *
import json
import faiss
import os

model_name = 'deepseek'
if model_name == 'llama':
    model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
elif model_name =='Qwen':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
elif model_name == 'deepseek':
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

#连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model = model.to(device)
departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科','中医科','急诊科','精神科','营养科','心理科','男科','其他科室']
departments_big = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科','中医科','其他科室']
departments_small = [ '传染科','急诊科','精神科','营养科','心理科','男科']
datapath = 'data/department_KB_' + model_name
for dep in departments:
    index_path = datapath + f"/{dep}/embeddings.index"
    map_path =datapath + f"/{dep}/mapping.json"

    # Check if both files exist
    if os.path.exists(index_path) and os.path.exists(map_path):
        print(f"{dep} already processed. Deleting existing files...")
        
        # 删除文件
        os.remove(index_path)
        os.remove(map_path)
    
    print(f"Deleted {index_path} and {map_path}.")

    # Determine depth based on department
    if dep in departments_big:
        depth = 3
    else:
        depth = 2

    start_time = time.time()
    
    # Ensure directory exists
    if not os.path.exists(datapath + f"/{dep}"):
        os.makedirs(datapath + f"/{dep}")

    print(dep + ' computing...')
    
    # Extract triples and compute embeddings
    triples = triple_to_text(extract_subgraph({'dep': [dep]}, graph, depth))
    KB_embeddings = get_sentence_embeddings_batch(triples,tokenizer,model, device)
    KB_embeddings = np.array(KB_embeddings).astype('float32')
    dimension = len(KB_embeddings[0])    
    M = 16
    ef_construction = 200
    
    # Create and configure FAISS index
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.ef_construction = ef_construction
    
    # Add embeddings to index
    index.add(KB_embeddings)
    
    # Save index and mapping
    faiss.write_index(index, index_path)
    
    mapping = {i: triple for i, triple in enumerate(triples)}
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    print(dep + ' finish.')
    
    end_time = time.time()
    timeRecord(start_time, end_time)


