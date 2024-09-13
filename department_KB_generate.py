'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-13 02:06:27
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-13 11:06:05
FilePath: /LLM-medical-KG/department_KB_generate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-13 02:06:26
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-13 02:06:27
FilePath: /LLM-medical-KG/department_KB_generate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
import torch
from cmner import *
import json
import faiss
import os

#连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科','中医科','急诊科','精神科','营养科','心理科','男科','其他科室']
departments_big = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科','中医科','其他科室']
departments_small = [ '传染科','急诊科','精神科','营养科','心理科','男科']
for dep in departments:
    if dep in departments_big:
        depth=3
    else:
        depth=2
    start_time = time.time()
    index_path = f"data/department_KB/{dep}"
    map_path = f"data/department_KB/{dep}"
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    
    print(dep + ' computing...')
    triples = triple_to_text(extract_subgraph({'dep': [dep]}, graph, depth))
    # KB_embeddings = get_entity_embeddings(triples, device)
    # KB_embeddings = np.array(KB_embeddings).astype('float32')
    # dimension = len(KB_embeddings[0])    
    # M = 16
    # ef_construction = 200
    
    # index = faiss.IndexHNSWFlat(dimension, M)
    # index.hnsw.ef_construction = ef_construction
    
    # # 添加嵌入到索引中
    # index.add(KB_embeddings)
    
    # # 保存索引到文件
    # faiss.write_index(index, index_path + "/embeddings.index")
    
    # 保存映射关系
    mapping = {i: triple for i, triple in enumerate(triples)}
    with open(map_path + "/mapping.json", 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    print(dep + ' finish.')
    end_time = time.time()
    timeRecord(start_time,end_time)
    pass


