
from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig
import time
import torch
from cmner import *
import json
import faiss
import os
from multi_agent import load_model_and_tokenizer



def process_department_embeddings(model_name, graph, tokenizer, model, device):
    datapath = 'data/department_KB_' + model_name

    departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科', '中医科', '急诊科', '精神科', '营养科', '心理科', '男科', '其他科室']
    departments_big = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '中医科', '其他科室']
    departments_small = ['传染科', '急诊科', '精神科', '营养科', '心理科', '男科']
    
    for dep in departments:
        index_path = datapath + f"/{dep}/embeddings.index"
        map_path = datapath + f"/{dep}/mapping.json"

        # 检查文件是否存在
        if os.path.exists(index_path) and os.path.exists(map_path):
            # print(f"{dep} already processed. Deleting existing files...")
            # # 删除文件
            # os.remove(index_path)
            # os.remove(map_path)
            # print(f"Deleted {index_path} and {map_path}.")
            print(f"{dep} already processed.continue...")
            continue

        # 根据部门确定深度
        depth = 3 if dep in departments_big else 2

        start_time = time.time()
        
        # 确保目录存在
        dep_dir = datapath + f"/{dep}"
        if not os.path.exists(dep_dir):
            os.makedirs(dep_dir)

        print(dep + ' computing...')
        
        # 提取三元组并计算嵌入
        triples = triple_to_text(extract_subgraph({'dep': [dep]}, graph, depth))
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

        print(dep + ' finish.')
        
        end_time = time.time()
        timeRecord(start_time, end_time)


if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    

    #连接到neo4j，获得知识图谱graph
    profile = "bolt://neo4j:Crj123456@localhost:7687"
    graph = Graph(profile)
    llms_name = [ 'glm']
    
    # 请确保 graph、tokenizer 和 model 已定义并初始化
    for model_name in llms_name:
        model,tokenizer = load_model_and_tokenizer(model_name,device)
        model = model.to(device)
        process_department_embeddings(model_name, graph, tokenizer, model, device)