import argparse
import numpy as np
import torch.optim
from transformers import AutoTokenizer, AutoModel,AutoModelForTokenClassification, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import faiss
import re
import os



class NER:
    """
    实体命名实体识别
    """
    def __init__(self,model_path) -> None:
        """
        Args:
            model_path:模型地址
        """

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

    def ner(self,sentence:str) -> list:
        """
        命名实体识别
        Args:
            sentence:要识别的句子
        Return:
            实体列表:[{'type':'LOC','tokens':[...]},...]
        """
        ans = []
        for i in range(0,len(sentence),500):
            ans = ans + self._ner(sentence[i:i+500])
        return ans
    
    def _ner(self,sentence:str) -> list:
        if len(sentence) == 0: return []
        inputs = self.tokenizer(
            sentence, add_special_tokens=True, return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda:0'))
            for key in inputs:
                inputs[key] = inputs[key].to(torch.device('cuda:0'))
            
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
        entities = []
        entity = {}
        for idx, token in enumerate(self.tokenizer.tokenize(sentence,add_special_tokens=True)):
            if 'B-' in predicted_tokens_classes[idx] or 'S-' in predicted_tokens_classes[idx]:
                if len(entity) != 0:
                    entities.append(entity)
                entity = {}
                entity['type'] = predicted_tokens_classes[idx].replace('B-','').replace('S-','')
                entity['tokens'] = [token]
            elif 'I-' in predicted_tokens_classes[idx] or 'E-' in predicted_tokens_classes[idx] or 'M-' in predicted_tokens_classes[idx]:
                if len(entity) == 0:
                    entity['type'] = predicted_tokens_classes[idx].replace('I-','').replace('E-','').replace('M-','')
                    entity['tokens'] = []
                entity['tokens'].append(token)
            else:
                if len(entity) != 0:
                    entities.append(entity)
                    entity = {}
        if len(entity) > 0:
            entities.append(entity)
        return entities

def save_to_md(file_name, content):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(content)

    # print(f"Content successfully saved to {file_name}")

def timeRecord(start_time, end_time):
    total_seconds = end_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if total_seconds < 60:
        print(f"运行时长: {seconds:.2f} 秒")
    elif total_seconds < 3600:
        print(f"运行时长: {int(minutes)} 分钟 {int(seconds)} 秒")
    else:
        print(f"运行时长: {int(hours)} 小时 {int(minutes)} 分钟 {int(seconds)} 秒")


def extract_entities(question):
    """
    使用 NER 模型从输入文本中提取实体，并返回所有识别出的词。
    
    参数:
    - question (str): 输入的文本字符串
    
    返回:
    - List[str]: 包含所有识别出的实体词的列表

    """
    ner_model = NER('/root/.cache/huggingface/hub/models--lixin12345--chinese-medical-ner/snapshots/5765a4d70ecf76d279d9f98bf4cdf0c52d388c7c')
    entities = ner_model.ner(question)  # 调用extract_entities函数
    entities = list(set([''.join(d.get('tokens', [])) for d in entities]))
    
    return entities

def get_entity_embeddings(entities, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    
    model.to(device)
    inputs = tokenizer(entities, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    
    # 取 [CLS] token
    embeddings = last_hidden_state[:, 0, :]

    return embeddings.cpu().numpy()

def get_entity_embeddings_batch(entities, device, batch_size=8):
    # 加载 BERT 模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.to(device)

    all_embeddings = []
    
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())
        print(f"{i}/{len(entities)}")

    return np.concatenate(all_embeddings, axis=0)

def get_sentence_embeddings_batch(texts, tokenizer, model, device, batch_size=8):
    all_embeddings = []

    # 将文本分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 编码输入
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # 启用 output_hidden_states=True
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 隐藏状态位于 outputs.hidden_states 中
        hidden_states = outputs.hidden_states  # List of tensors (layers)
        
        # 提取最后一层隐藏状态的 CLS token
        cls_embedding = hidden_states[-1][:, 0, :]  # 最后一层的第一个 token (CLS token)
        
        # 将结果添加到总 embedding 列表中
        all_embeddings.append(cls_embedding.cpu().numpy())
        print(f"{i}/{len(texts)}")


    # 合并所有批次的 embedding
    return np.concatenate(all_embeddings, axis=0)


def get_relative_nodenames(entities, device, k=2):
    """
    对于提取的实体，使其与图中的节点对齐，返回图中的相关节点
    """
    embeddings = get_entity_embeddings(entities, device)
    print(embeddings.shape)  

    index_with_ids = faiss.read_index('data/node_embeddings.index')

    distances, indices = index_with_ids.search(embeddings, k)

    relative_nodenames = {}

    with open("data/dict_nodes.json", "r", encoding='utf-8') as f:
        id_to_name = json.load(f)

    for i, idx_list in enumerate(indices):
        print(f"Query: {entities[i]}")
        relative_nodenames[entities[i]] = []
        for j, idx in enumerate(idx_list):
            nearest_name = id_to_name[str(idx)]  # 注意 id 是字符串类型
            relative_nodenames[entities[i]].append(nearest_name)
            print(f"Nearest Neighbor {j+1}: {nearest_name} (ID: {idx})")
            print(f"Distance: {distances[i][j]}")
        print()

    return relative_nodenames


def extract_subgraph(relative_nodenames, graph, depth = 1):
    """
    对于图中的节点，返回指定阶数的子图，保留关系方向。
    """
    subgraphs = {}

    for entity, rel_nodes in relative_nodenames.items():
        for node in rel_nodes:
            query = f"""
            MATCH (n {{name: '{node}'}}) 
            OPTIONAL MATCH path = (n)-[r*1..{depth}]-(m) 
            WITH relationships(path) AS rels, nodes(path) AS nodes
            WHERE size(rels) > 0
            UNWIND range(0, size(rels)-1) AS idx
            WITH nodes[idx] AS source_node, nodes[idx+1] AS target_node, 
                rels[idx].name AS relationship, 
                startNode(rels[idx]) AS start_node, endNode(rels[idx]) AS end_node
            WHERE source_node IS NOT NULL AND target_node IS NOT NULL AND relationship IS NOT NULL
            WITH 
                CASE 
                    WHEN start_node <> source_node THEN target_node
                    ELSE source_node
                END AS final_source_node,
                CASE 
                    WHEN start_node <> source_node THEN source_node
                    ELSE target_node
                END AS final_target_node,
                relationship
            RETURN DISTINCT 
                final_source_node.name AS source_node, 
                relationship AS relationship, 
                final_target_node.name AS target_node
            """
            result = graph.run(query).data()
            print(f"Subgraph of {node} (from {entity}) finished.")
            subgraphs[node] = result

    return subgraphs




def triple_to_text(subgraphs):
    """ 
    将取得的子图三元组转化为文本
    """
    generated_texts = []

    for key, triples in subgraphs.items():
        for triple in triples:
            source = triple['source_node']
            relationship = triple['relationship']
            target = triple['target_node']

            # 根据不同的关系类型生成自然语言描述
            if relationship == "症状":
                generated_texts.append(f"{source}可能表现{target}症状。")
            elif relationship == "并发症":
                generated_texts.append(f"{source}可能有{target}的并发症。")
            elif relationship == "推荐食谱":
                generated_texts.append(f"患有{source}时，推荐的食谱包括{target}。")
            elif relationship == "忌吃":
                generated_texts.append(f"患有{source}时，应避免食用{target}。")
            elif relationship == "宜吃":
                generated_texts.append(f"患有{source}时，建议食用{target}。")
            elif relationship == "属于":
                generated_texts.append(f"{source}属于{target}。")
            elif relationship == "常用药品":
                generated_texts.append(f"治疗{source}常用的药品有{target}。")
            elif relationship == "生产药品":
                generated_texts.append(f"{source}生产的药品包括{target}。")
            elif relationship == "好评药品":
                generated_texts.append(f"治疗{source}的好评药品包括{target}。")
            elif relationship == "诊断检查":
                generated_texts.append(f"诊断{source}可以通过{target}进行检查。")
            elif relationship == "所属科室":
                generated_texts.append(f"{source}属于{target}。")
            else:
                generated_texts.append(f"{source}与{target}的关系是{relationship}。")

    return generated_texts

def pruning(subgraphs, question, device, top_n=None, similarity_threshold=None):
    """
    剪枝：将三元组转化的文本的embedding与query的embedding进行相似度匹配，保留相似度高的内容。
    """
    texts_from_subgraphs = triple_to_text(subgraphs)
    question_embedding = get_entity_embeddings([question], device)[0]
    texts_embeddings = get_entity_embeddings(texts_from_subgraphs,device)
    similarities = cosine_similarity([question_embedding], texts_embeddings)[0]
    sorted_indices = similarities.argsort()[::-1]  # 从高到低排序
    texts_relative = []

    if top_n is not None:
        # 保留相似度最高的top_n个文本
        texts_relative = [texts_from_subgraphs[i] for i in sorted_indices[:top_n]]
    elif similarity_threshold is not None:
        # 保留相似度超过阈值的文本
        texts_relative = [texts_from_subgraphs[i] for i in sorted_indices if similarities[i] >= similarity_threshold]
    else:
        raise ValueError("You must specify either top_n or similarity_threshold.")
    # save_to_md('test_outputs.md','\nquestion相关内容\n')
    # save_to_md('test_outputs.md','\n'.join(texts_relative) + '\n')

    return texts_relative
    

def generate_subgraphs(question, graph,device):
    entities = extract_entities(question)
    relative_nodenames = get_relative_nodenames(entities, device)
    subgraphs = extract_subgraph(relative_nodenames,graph)
    subgraphs = pruning(subgraphs, question, device, top_n =50)
    print(subgraphs)

    return subgraphs


def extract_depKB(question, dep,tokenizer,model, device,model_name,top_n=20):
    print(f"{dep}科室检索知识中...")
    base_path = '/root/LLM-medical-KG/data/department_KB_' + model_name
    dep_path = os.path.join(base_path, dep)
    index_file = os.path.join(dep_path, 'embeddings.index')
    mapping_file = os.path.join(dep_path, 'mapping.json')
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    question_embedding = get_sentence_embeddings_batch([question],tokenizer,model, device)

    index = faiss.read_index(index_file)
    D, I = index.search(np.array(question_embedding).astype(np.float32), top_n)
    
    top_n_strings = [mapping[str(i)] for i in I[0]]
    for string in top_n_strings:
        print(string)
    return top_n_strings
