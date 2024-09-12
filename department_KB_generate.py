from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
import torch
from cmner import *
import json
import faiss

#连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科','中医科','急诊科','精神科','营养科','心理科','男科','其他科室']

for dep in departments:
    print(dep + ' computing...')
    triples = triple_to_text(extract_subgraph({'dep': [dep]}, graph, 3))
    KB_embeddings = get_entity_embeddings(triples, device)
    KB_embeddings = np.array(KB_embeddings).astype('float32')
    dimension = len(KB_embeddings[0])
    
    M = 16
    ef_construction = 200
    
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.ef_construction = ef_construction
    
    # 添加嵌入到索引中
    index.add(KB_embeddings)
    
    # 保存索引到文件
    faiss.write_index(index, f"data/department_KB/{dep}/embeddings.index")
    
    # 保存映射关系
    mapping = {i: triple for i, triple in enumerate(triples)}
    with open(f"data/department_KB/{dep}/mapping.json", 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    print(dep + ' finish.')



