from cmner import *
from py2neo import Graph
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datetime import datetime
import os
from dep.get_departments import get_dep

# 定义一些文件和路径
file_name = "test_outputs.md"
profile = "bolt://neo4j:Crj123456@localhost:7687"

departments,_ = get_dep()

# 检查 results 文件夹是否存在
if not os.path.exists("results"):
    os.makedirs("results")

# 连接到 neo4j
def connect_to_graph():
    graph = Graph(profile)
    return graph

# 加载模型和 tokenizer
def load_model_and_tokenizer(model_name,device):
    if model_name == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3.1-8B-Chinese-Chat")
        model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3.1-8B-Chinese-Chat")
    elif model_name == 'Qwen':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    elif model_name == 'deepseek':
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat")
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif model_name == 'glm':
        tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/glm-4-9b-chat",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()
    elif model_name == 'baichuan':
        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
            use_fast=False,
            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
            torch_dtype=torch.float16,
            trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")
    elif model_name == 'huatuo':
        model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", trust_remote_code=True)

    elif model_name == 'baichuan':
        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
        
        
    model.to(device)
    return model, tokenizer

class MedicalAgent:
    def __init__(self, department_name, model,model_name, tokenizer, device):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
    def generate_response(self, query):
        # 使用 extract_depKB 函数获取与 query 匹配的科室知识
        top_n_strings = extract_depKB(query, self.department_name, self.tokenizer, self.model, self.device, self.model_name, top_n=10)
        # 将知识库的字符串列表拼接成一段提示
        kb_prompt = "\n".join([f"{i+1}. {s}" for i, s in enumerate(top_n_strings)])
        # 构造包含科室知识和患者信息的 prompt
        prompt = f"你是一名经验丰富的{self.department_name}专家，请根据以下患者信息提供专业意见：\n患者信息：{query}\n与该问题相关的知识：\n{kb_prompt}\n你的发言："
        print(f"{self.department_name}发言中...")

        # 生成科室的回答
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=500, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

        # 保存发言内容
        save_to_md(file_name, f"\n-------{self.department_name}专家发言--------\n" + response)

        return response

class LeaderAgent:
    def __init__(self, model, tokenizer, agents, graph, device):
        self.model = model  
        self.tokenizer = tokenizer
        self.agents = agents  
        self.graph = graph 
        self.device = device

    def consult(self, query):
        save_to_md(file_name, '\nquery:\n' + query + '\n')
        relevant_agents = self.decide_agents_via_leader(query)
        knowledge_subgraphs = generate_subgraphs(query, self.graph, self.device)
        responses = self.collect_responses(relevant_agents, query)
        combined_responses = self.combine_responses_with_knowledge(responses, knowledge_subgraphs)
        final_response = self.summarize_with_leader_agent(combined_responses)
        return final_response

    def decide_agents_via_leader(self, query):
        extracted_departments = departments  # 暂时不要科室选择过程
        save_to_md(file_name, f"\n------departments------\n" + ','.join(extracted_departments) + "\n-----------------------\n")
        return [agent for agent in self.agents.values() if agent.department_name in extracted_departments]

    def collect_responses(self, agents, query):
        responses = {}
        for agent in agents:
            response = agent.generate_response(query)
            responses[agent.department_name] = response
        return responses

    def combine_responses_with_knowledge(self, responses, knowledge_subgraphs):
        combined = ""
        for department, response in responses.items():
            combined += f"【{department}的建议】\n{response}\n"
        combined += "\n【相关的知识图谱内容】\n" + "\n".join(knowledge_subgraphs)
        return combined

    def summarize_with_leader_agent(self, combined_responses):
        prompt = f"以下是各科室医生的会诊结果和相关知识：\n{combined_responses}\n请根据这些内容生成最终诊断总结。你的总结应当参考各个科室专家的建议，做到尽可能的全面，并排除你认为有误的信息。你的总结："
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id)
        final_response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return final_response

# 封装整个咨询流程的主函数
def run_medical_consultation(query, model_name='deepseek',device = "cuda" if torch.cuda.is_available() else "cpu"):
    # 1. 连接知识图谱
    graph = connect_to_graph()
    
    # 2. 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name,device)

    # 3. 初始化科室智能体
    agents = {dep: MedicalAgent(dep, model,model_name, tokenizer, device) for dep in departments}

    # 4. 初始化 LeaderAgent
    leader_agent = LeaderAgent(model, tokenizer, agents, graph, device)

    # 5. 执行咨询流程
    start_time = time.time()
    final_answer = leader_agent.consult(query)
    print(f"最终问诊结果: {final_answer}")
    save_to_md(file_name, f"\n---------------\n最终问诊结果:\n {final_answer}")
    end_time = time.time()
    timeRecord(start_time, end_time)

    return final_answer
