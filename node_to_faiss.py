from py2neo import Graph
import json
import faiss
import torch
import os
from cmner import *

# 连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 查询所有Disease节点
query = "MATCH (n:Disease) RETURN n"
result = graph.run(query)

# 初始化字典，分别存储每个属性
properties_dict = {
    "name": {},
    "desc": {},
    "prevent": {},
    "cure_way": {},
    "cure_lasttime": {},
    "cured_prob": {},
    "cause": {},
    "cure_department": {},
    "easy_get": {}
}

# 遍历结果，将各属性分开保存
for record in result:
    node = record['n']
    node_id = node.identity  # 获取节点ID

    # 直接从Node对象访问属性
    properties_dict["name"][node_id] = node.get('name', "")
    properties_dict["desc"][node_id] = node.get('desc', "")
    properties_dict["prevent"][node_id] = node.get('prevent', "")
    properties_dict["cure_way"][node_id] = node.get('cure_way', "")
    properties_dict["cure_lasttime"][node_id] = node.get('cure_lasttime', "")
    properties_dict["cured_prob"][node_id] = node.get('cured_prob', "")
    properties_dict["cause"][node_id] = node.get('cause', "")
    properties_dict["cure_department"][node_id] = node.get('cure_department', "")
    properties_dict["easy_get"][node_id] = node.get('easy_get', "")

# 创建文件夹来保存属性的JSON文件
output_dir = "data/properties"
os.makedirs(output_dir, exist_ok=True)

# 将每个属性的字典分别保存为单独的JSON文件
for prop, data in properties_dict.items():
    filename = os.path.join(output_dir, f"dict_node_{prop}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# 获取节点名称用于embedding
node_names = list(properties_dict["name"].values())

# 获取节点的embedding
all_embeddings = get_entity_embeddings(node_names, device)

dimension = all_embeddings.shape[1]
M = 16
ef_construction = 200

# 创建 HNSW 索引
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.ef_construction = ef_construction

# 使用 IndexIDMap 来存储 ID
index_with_ids = faiss.IndexIDMap(index)

# 将 embeddings 和 ids 添加到索引中
ids = list(map(int, properties_dict["name"].keys()))
index_with_ids.add_with_ids(all_embeddings, ids)

# 保存索引到文件
faiss.write_index(index_with_ids, "data/node_disease_embeddings.index")
