'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-11 07:52:53
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-11 10:09:44
FilePath: /LLM-medical-KG/node_to_faiss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# query = "MATCH (n) RETURN n.name"
# result = graph.run(query)
# all_entities = [record['n.name'] for record in result]
# dict_nodes = {i:all_entities[i] for i in range(len(all_entities))}

# # 将元数据保存到JSON文件
# with open("dict_nodes.json", "w", encoding="utf-8") as f:
#     json.dump(dict_nodes, f,ensure_ascii=False)

# 从JSON文件加载元数据
with open("data/dict_nodes.json", "r",encoding='utf-8') as f:
    loaded_data = json.load(f)
ids = list(map(int, loaded_data.keys()))
node_names = list(loaded_data.values())
all_embeddings = get_entity_embeddings(node_names, device)


dimension = 768
M = 16
ef_construction = 200

# 创建 HNSW 索引
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.ef_construction = ef_construction


# 使用 IndexIDMap 来存储 ID
index_with_ids = faiss.IndexIDMap(index)  # 包装 index 以便存储 ID

# 将 embeddings 和 ids 添加到索引中
index_with_ids.add_with_ids(all_embeddings, ids)  # 将向量和对应的 ID 添加到索引中

# 保存索引到文件
faiss.write_index(index_with_ids, "data/node_embeddings_bert.index")  # 保存索引文件