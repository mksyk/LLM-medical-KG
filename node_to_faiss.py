from py2neo import Graph
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from cmner import *
import json
import faiss

# #连接到neo4j，获得知识图谱graph
# profile = "bolt://neo4j:Crj123456@localhost:7687"
# graph = Graph(profile)

#加载大模型model
# model_name_or_path = "/root/.cache/huggingface/hub/models--FreedomIntelligence--HuatuoGPT2-7B/snapshots/1490cc91a93d2d0d2fdc9d3681bc1c5099cde163" #huatuoGPT2
model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e" #llama3.1-8B chinese


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
model.config.pad_token_id = tokenizer.eos_token_id

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
all_embeddings = get_entity_embeddings(node_names, model, tokenizer, device)


dimension = 4096
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
faiss.write_index(index_with_ids, "data/node_embeddings_ids.index")  # 保存索引文件