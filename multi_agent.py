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
        tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", trust_remote_code=True)

    elif model_name == 'baichuan':
        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)

    elif model_name == 'Qwen2':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
        
        
    model.to(device)
    return model, tokenizer


def get_original_model_output(query, model,tokenizer, device):
    print('Getting original model output...')
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(query):].strip()
    return response


class MedicalAgent:
    def __init__(self, department_name, model, tokenizer, device):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, query):
        # 使用 extract_depKB 函数获取与 query 匹配的科室知识
        top_n_strings = extract_depKB(query, self.department_name, self.tokenizer, self.model, self.device, top_n=10)

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
    def __init__(self, model, tokenizer, graph, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.graph = graph
        self.device = device

        # 科室预测模型配置
        model_name = "/root/.cache/huggingface/hub/models--uer--sbert-base-chinese-nli/snapshots/2081897a182fdc33ea6e840f0eb38959b63ec0d3"  # 训练时使用的模型名称
        self.dep_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dep_transformer_model = AutoModel.from_pretrained(model_name).to(device)

        input_dim = 768  # 对应 SBERT/BERT 输出维度
        hidden_dim = 512  # 隐藏层维度
        self.departments, _ = get_dep()  # 获取科室列表
        output_dim = len(self.departments)

        # 初始化分类器模型
        self.dep_classifier = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        self.dep_classifier.load_state_dict(torch.load("dep_model/dep_classifier.pth"))
        self.dep_classifier.eval()
        self.dep_transformer_model.eval()

    def consult(self, query):
        save_to_md(file_name, '\nquery:\n' + query + '\n')

        # 1. 通过科室分类器决定需要的科室
        relevant_departments = list(self.predict_departments(query).keys())
        save_to_md(file_name, f"\n------departments------\n" + ','.join(relevant_departments) + "\n-----------------------\n")

        # 2. 动态初始化对应的 MedicalAgent
        relevant_agents = self.initialize_medical_agents(relevant_departments)

        # 3. 收集 MedicalAgent 的响应
        responses = self.collect_responses(relevant_agents, query)

        # 4. 整合响应和知识
        knowledge_subgraphs = generate_subgraphs(query, self.graph, self.device)
        combined_responses = self.combine_responses_with_knowledge(responses, knowledge_subgraphs)

        # 5. 生成最终的诊断总结
        final_response = self.summarize_with_leader_agent(combined_responses)
        ori_model_output = get_original_model_output(query,self.model,self.tokenizer,self.device)
        return final_response,ori_model_output

    def predict_departments(self, query):
        # 使用科室分类器预测相关的科室
        return predict_department(query, self.dep_classifier, self.dep_tokenizer, self.dep_transformer_model, self.departments, self.device)

    def initialize_medical_agents(self, relevant_departments):
        # 动态初始化每个相关科室的 MedicalAgent
        return [MedicalAgent(dep, self.model, self.tokenizer, self.device) for dep in relevant_departments]

    def collect_responses(self, agents, query):
        # 收集每个 MedicalAgent 的响应
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


# 主流程函数
def run_medical_consultation(query, model_name='deepseek', device="cuda" if torch.cuda.is_available() else "cpu"):
    # 1. 连接知识图谱
    graph = connect_to_graph()
    
    # 2. 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # 3. 初始化 LeaderAgent
    leader_agent = LeaderAgent(model, tokenizer, graph, device)

    # 4. 执行咨询流程
    start_time = time.time()
    final_answer,ori_model_output = leader_agent.consult(query)
    print(f"最终问诊结果: {final_answer}")
    save_to_md(file_name, f"\n---------------\n最终问诊结果:\n {final_answer}")
    end_time = time.time()
    timeRecord(start_time, end_time)

    return final_answer,ori_model_output