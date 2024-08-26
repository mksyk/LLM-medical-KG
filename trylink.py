from py2neo import Graph
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from cmner import extract_entities
import torch

start_time = time.time()

#连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)

#加载大模型model
# model_name_or_path = "/root/.cache/huggingface/hub/models--FreedomIntelligence--HuatuoGPT2-7B/snapshots/1490cc91a93d2d0d2fdc9d3681bc1c5099cde163" #huatuoGPT2
model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e" #llama3.1-8B chinese


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
model.config.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def query_knowledge_graph(graph, query):
    query_result = graph.run(query).data()
    print("|-------query_result:")
    print(query_result)

    knowledge = " ".join([str(item) for item in query_result])
    return knowledge

def generate_answer_with_rag(question, graph, model, tokenizer):
    entities = extract_entities(question)
    print(entities)
    conditions = " OR ".join([f"n.name CONTAINS '{entity}'" for entity in entities])
    query = f"MATCH (n) WHERE {conditions} WITH n LIMIT 1 OPTIONAL MATCH (n)-[*1]-(m) RETURN n, collect(m) AS neighbors"
    knowledge = query_knowledge_graph(graph, query)
    print("|-------knowledge:")
    print(knowledge)
    
    input_text = f"这里有一些你可以参考的知识：{knowledge}，请你以此信息帮助解决以下病人的问题： {question}"
    # input_text = question
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

question = "我有一些咳嗽和流鼻涕的症状，请问你有什么建议？"
answer = generate_answer_with_rag(question, graph, model, tokenizer)
print("---answer---")
print(answer)

end_time = time.time()
runtime = end_time - start_time
print(f"运行时间: {runtime:.2f} 秒")

