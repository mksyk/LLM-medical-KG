'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-26 09:45:00
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-26 09:45:00
FilePath: /LLM-medical-KG/depKB_sbert.py
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
from multi_agent import load_model_and_tokenizer
from dep.get_departments import get_dep



def process_department_embeddings(graph, tokenizer, model, device):
    datapath = 'data/department_KB'

    departments,depths = get_dep()
    for i in range(len(departments)):
        index_path = datapath + f"/{departments[i]}/embeddings.index"
        map_path = datapath + f"/{departments[i]}/mapping.json"

        # 检查文件是否存在
        if os.path.exists(index_path) and os.path.exists(map_path):
            # print(f"{departments[i]} already processed. Deleting existing files...")
            # # 删除文件
            # os.remove(index_path)
            # os.remove(map_path)
            # print(f"Deleted {index_path} and {map_path}.")
            continue
            print(f"{departments[i]} already processed.continue...")

        # 根据部门确定深度
        depth = depths[i]

        start_time = time.time()
        
        # 确保目录存在
        dep_dir = datapath + f"/{departments[i]}"
        if not os.path.exists(dep_dir):
            os.makedirs(dep_dir)

        print(departments[i] + ' computing...')
        
        # 提取三元组并计算嵌入
        triples = triple_to_text(extract_subgraph({'dep': [departments[i]]}, graph, depth))
        KB_embeddings = get_sentence_embeddings_batch(triples, tokenizer, model, device)
        KB_embeddings = np.array(KB_embeddings).astype('float32')
        dimension = len(KB_embeddings[0])    
        M = 16
        ef_construction = 200
        
        # 创建并配置 FAISS 索引
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.ef_construction = ef_construction
        
        # 将嵌入添加到索引
        index.add(KB_embeddings)
        
        # 保存索引和映射
        faiss.write_index(index, index_path)
        
        mapping = {i: triple for i, triple in enumerate(triples)}
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)

        print(departments[i] + ' finish.')
        
        end_time = time.time()
        timeRecord(start_time, end_time)

# 示例调用
if __name__ == "__main__": 
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    

    #连接到neo4j，获得知识图谱graph
    profile = "bolt://neo4j:Crj123456@localhost:7687"
    graph = Graph(profile)
   
    tokenizer = AutoTokenizer.from_pretrained("uer/sbert-base-chinese-nli")
    model = AutoModel.from_pretrained("uer/sbert-base-chinese-nli")
    model = model.to(device)
    process_department_embeddings(graph, tokenizer, model, device)