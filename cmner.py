import argparse
import numpy as np
import pytorch_lightning as pl
import torch.optim
from transformers import AutoTokenizer,BertTokenizerFast, get_linear_schedule_with_warmup, \
    AutoModelForTokenClassification
from sklearn.metrics.pairwise import cosine_similarity
import json
import faiss
import re

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

# class MedicalNerModel(pl.LightningModule):

#     def __init__(self, args: argparse.Namespace):
#         super(MedicalNerModel, self).__init__()
#         self.args = args
#         self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
#         self.model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=5)

#         self.val_correct_num = 0
#         self.val_total_num = 0

#     def training_step(self, batch, batch_idx, *args, **kwargs):
#         inputs, targets, = batch
#         outputs = self.model(**inputs, labels=targets)
#         loss = outputs.loss
#         outputs = outputs.logits

#         self.log("train_loss", loss.item(), prog_bar=True)

#         return {
#             'loss': loss,
#             'outputs': outputs.argmax(-1) * inputs['attention_mask'],
#             'targets': targets,
#         }

#     def on_train_batch_end(self, outputs, batch, batch_idx: int):
#         targets_size = batch[1].size()
#         preds = outputs['outputs']
#         targets = outputs['targets']

#         correct_num = torch.all(preds == targets, dim=1).sum().item()
#         total_num = targets_size[0]

#         self.log("train_acc", correct_num / total_num, prog_bar=True)

#     def validation_step(self, batch, batch_idx, *args, **kwargs):
#         inputs, targets = batch
#         outputs = self.model(**inputs).logits

#         preds = outputs.argmax(-1) * inputs['attention_mask']

#         correct_num = torch.all(preds == targets, dim=1).sum().item()
#         total_num = targets.size(0)

#         self.log("val_acc", correct_num / total_num)

#         self.val_correct_num += correct_num
#         self.val_total_num += total_num

#         return {
#             'outputs': preds,
#             'targets': targets,
#         }

#     def on_validation_epoch_end(self) -> None:
#         print("Epoch",self.current_epoch, ". val_acc:", self.val_correct_num / self.val_total_num)
#         self.val_correct_num = 0
#         self.val_total_num = 0

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)

#         t_total = len(self.args.train_loader) * self.args.epochs

#         warmup_steps = int(0.1 * t_total)
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
#         )

#         return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
   
#     @staticmethod
#     def format_outputs(sentences, outputs):
#         preds = []
#         for i, pred_indices in enumerate(outputs):
#             words = []
#             start_idx = -1
#             end_idx = -1
#             flag = False
#             for idx, pred_idx in enumerate(pred_indices):
#                 if pred_idx == 1:
#                     start_idx = idx
#                     flag = True
#                     continue

#                 if flag and pred_idx != 2 and pred_idx != 3:
#                     # 出现了不应该出现的index
#                     # print("Abnormal prediction results for sentence", sentences[i])
#                     start_idx = -1
#                     end_idx = -1
#                     continue

#                 if pred_idx == 3:
#                     end_idx = idx

#                     words.append({
#                         "start": start_idx,
#                         "end": end_idx + 1,
#                         "word": sentences[i][start_idx:end_idx+1]
#                     })
#                     start_idx = -1
#                     end_idx = -1
#                     flag = False
#                     continue

#             preds.append(words)

#         return preds


def remove_punctuation_and_newlines(text):
    """
    删除输入字符串中的所有换行符和标点符号（包括中英文）。
    
    参数:
    - text (str): 输入的字符串
    
    返回:
    - str: 处理后的字符串
    """
    pattern = r"[^\w\s]"
    cleaned_text = re.sub(pattern, "", text)  
    cleaned_text = cleaned_text.replace("\n", "")
    cleaned_text = cleaned_text.replace("\r", "") 
    
    return cleaned_text


def extract_entities(question):
    """
    使用 NER 模型从输入文本中提取实体，并返回所有识别出的词。
    
    参数:
    - question (str): 输入的文本字符串
    
    返回:
    - List[str]: 包含所有识别出的实体词的列表

    """
    question = remove_punctuation_and_newlines(question)
    ner_model = NER('lixin12345/chinese-medical-ner')
    entities = ner_model.ner(question)  # 调用extract_entities函数
    entities = list(set([''.join(d.get('tokens', [])) for d in entities]))
    
    return entities

    
def get_entity_embeddings(entities, model, tokenizer, device):
    """
    使用LLM计算取得的实体的embedding
    """
    embeddings = []
    count = 0 
    for entity in entities:
        inputs = tokenizer(entity, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1).cpu().detach().numpy()
        embeddings.append(embedding)
        count += 1
        # print(f"{count} : embedding of {entity} finish.")
    return np.vstack(embeddings)

def get_question_embbeding(query, model, tokenizer, device):
    embeddings = []
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    return last_hidden_state.mean(dim=1).cpu().detach().numpy()


def get_relative_nodenames(entities, model, tokenizer, device,k=2):
    """
    对于提取的实体，使其与图中的节点对齐，返回图中的相关节点
    """
    embeddings = get_entity_embeddings(entities, model, tokenizer, device)
    index_with_ids = faiss.read_index('data/node_embeddings_ids.index')

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
        #     print(f"Nearest Neighbor {j+1}: {nearest_name} (ID: {idx})")
        #     print(f"Distance: {distances[i][j]}")
        # print()

    return relative_nodenames

def extract_subgraph(relative_nodenames, graph, depth = 1):
    """
    对于图中的节点，返回指定阶数的子图。
    """
    subgraphs = {}

    for entity, rel_nodes in relative_nodenames.items():
        for node in rel_nodes:
            query = f"""
            MATCH (n {{name: '{node}'}}) 
            OPTIONAL MATCH path=(n)-[r*1..{depth}]-(m) 
            WITH n, relationships(path) AS rels, nodes(path) AS nodes
            WHERE size(rels) > 0
            UNWIND range(0, size(rels)-1) AS idx
            RETURN DISTINCT nodes[idx].name AS source_node, rels[idx].name AS relationship, nodes[idx+1].name AS target_node
            """
            result = graph.run(query).data()
            # print(f"Subgraph of {node} (from {entity}) finished.")
            subgraphs[node] = result
    return subgraphs



def triple_to_text(subgraphs):

    """
    Args:
        subgraphs (list[list]):每一行代表一个起始结点
    Returns:
        list[list]: list[起始结点] = 起始结点对应子图的自然语言形式 
    """
    final_texts = []

    for source_node, triples in subgraphs.items():
        generated_texts = []
        for triple in triples:
            source = triple['source_node']
            relationship = triple['relationship']
            target = triple['target_node']

            if relationship == "症状":
                generated_texts.append(f"{source}有可能是{target}的症状。")
            elif relationship == "并发症":
                generated_texts.append(f"{source}可能是{target}的并发症。")
            elif relationship == "推荐食谱":
                generated_texts.append(f"{source}的推荐食谱包括{target}。")
            elif relationship == "忌吃":
                generated_texts.append(f"{source}时忌吃{target}。")
            elif relationship == "宜吃":
                generated_texts.append(f"{source}时宜吃{target}。")
            elif relationship == "属于":
                generated_texts.append(f"{source}属于{target}。")
            elif relationship == "常用药品":
                generated_texts.append(f"{source}的常用药品是{target}。")
            elif relationship == "生产药品":
                generated_texts.append(f"{source}生产的药品包括{target}。")
            elif relationship == "好评药品":
                generated_texts.append(f"{source}的好评药品包括{target}。")
            elif relationship == "诊断检查":
                generated_texts.append(f"{source}可以通过{target}来诊断检查。")
            elif relationship == "所属科室":
                generated_texts.append(f"{source}属于{target}科室。")
            else:
                generated_texts.append(f"{source}与{target}的关系是{relationship}。")
        final_texts.append(generated_texts)
    # # 打印生成的文本列表
    # for text in generated_texts:
    #     print(text)
    return final_texts

def triple_to_text_triple(subgraphs):
    """ 
    将取得的子图三元组转化为文本形式，如
    """
    graph_changed_to_texts = []

    for key, triples in subgraphs.items():
        for triple in triples:
            source = triple['source_node']
            relationship = triple['relationship']
            target = triple['target_node']
            graph_changed_to_texts.append(f"{{{source}}},{{{relationship}}},{{{target}}}")
        

    return graph_changed_to_texts


def pruning(subgraphs, question, model, tokenizer, device, top_n=None, similarity_threshold=None):
    """
    剪枝：将三元组转化的文本的embedding与query的embedding进行相似度匹配，保留相似度高的内容。
    """
    texts_from_subgraphs = triple_to_text(subgraphs)
    question_embedding = get_question_embbeding(question, model, tokenizer, device)
    texts_embeddings = get_entity_embeddings(texts_from_subgraphs, model, tokenizer, device)
    similarities = cosine_similarity([question_embedding], texts_embeddings)[0]
    sorted_indices = similarities.argsort()[::-1]  # 从高到低排序
    texts_relative = []
    subgraph_relative = []
    if top_n is not None:
        # 保留相似度最高的top_n个文本
        #texts_relative = [texts_from_subgraphs[i] for i in sorted_indices[:top_n]]
        subgraph_relative = [subgraphs[i] for i in sorted_indices[:top_n]]
    elif similarity_threshold is not None:
        # 保留相似度超过阈值的文本
        #texts_relative = [texts_from_subgraphs[i] for i in sorted_indices if similarities[i] >= similarity_threshold]
        subgraph_relative = [subgraphs[i] for i in sorted_indices if similarities[i] >= similarity_threshold]
    else:
        raise ValueError("You must specify either top_n or similarity_threshold.")
    return subgraph_relative

    #return texts_relative
    
    
def pruning_more_than_one_hop_g(subgraphs, question, model, tokenizer,entities,device, top_n=None, similarity_threshold=None):
    """
    剪枝：将三元组转化的文本的embedding与query的embedding进行相似度匹配，保留相似度高的内容。
    """
    texts_from_subgraphs = triple_to_text(subgraphs)
    question_embedding = get_question_embbeding(question, model, tokenizer, device)
    entities_embbeding = get_entity_embeddings(entities, model, tokenizer, device)
    texts_embeddings = get_entity_embeddings(texts_from_subgraphs, model, tokenizer, device)
    #for subgraph in subgraphs:
        #对于每个子图
        #对于每个实体生成的多个子图，从最深度遍历，深度循环为 2-max_depth，假设max深度为4
        #对于当前子图的末尾entity与输入的所有entities比较相似度，当高于某一阈值时，则保留，否则删除
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

    return texts_relative
    

def generate_subgraphs(question, graph, model, tokenizer,device,leader = False):
    entities = extract_entities(question)
    relative_nodenames = get_relative_nodenames(entities, model, tokenizer, device)#可能情况，有irrelevant节点，需剪枝
    if leader:
        subgraphs = extract_subgraph(relative_nodenames,graph)
        subgraphs = pruning(subgraphs, question, model, tokenizer, device, top_n =50)#得到relative graph
    else:
        #没有用到 这里是准备传给科室agent时，提取科室节点的三阶子图
        subgraphs = extract_subgraph(relative_nodenames,graph,3)
        subgraphs = pruning(subgraphs, question, model, tokenizer, device, top_n =50)
    print(subgraphs)

    return subgraphs
