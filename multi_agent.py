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


def get_original_model_output(query, model, tokenizer, device):
    """
    获取原始模型的输出，prompt 中加入知识三元组信息，并与系统流程使用相同的生成逻辑。
    """
    prompt = f"你是一名经验丰富的医疗专家，请根据以下患者信息提供专业意见：\n患者信息：{query}\n你的诊断结果："

    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    save_to_md(file_name, f"\n-------原始模型输出--------\n{response}\n")
    
    return response


class MedicalAgent:
    def __init__(self, department_name, model, tokenizer, device):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, query):
        """
        MedicalAgent生成科室诊断结果，包含科室知识，并构建合理的prompt。
        """
        top_n_strings = extract_depKB(query, self.department_name, self.tokenizer, self.model, self.device, top_n=10)
        kb_prompt = "\n".join([f"{i+1}. {s}" for i, s in enumerate(top_n_strings)])

        prompt = (
            f"你是一名在{self.department_name}领域有多年经验的资深专家，精通处理复杂病例。"
            f"现在有一位患者向你咨询，以下是他的病情描述和相关的科室知识。知识的形式是三元组，实体-关系-实体（用逗号分隔）。如果你认为这些知识可以用于回答，你应该在回答中提到这些内容。"
            f"请基于这些信息提供你的专业诊断建议，结合你的丰富经验，给出可能的病因、进一步的检查建议，"
            f"以及你认为有必要时应采取的治疗方案。\n"
            f"患者信息：{query}\n"
            f"相关科室知识：\n{kb_prompt}\n"
            f"请详细回答："
        )

        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=500, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

        # 保存科室专家的发言
        # save_to_md(file_name, f"\n-------{self.department_name}专家发言--------\n{response}\n")

        return response


class LeaderAgent:
    def __init__(self, model, tokenizer, graph, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.graph = graph
        self.device = device
        self.departments, _ = get_dep()



    def consult(self, query):
        """
        LeaderAgent负责组织整个会诊流程，整合科室专家诊断与知识图谱，并最终生成总结。
        """
        save_to_md(file_name, f"\nquery:\n{query}\n")

        dep_chosen_result = self.predict_departments(query)
        save_to_md(file_name,f'\n\n------相关科室------\n{dep_chosen_result}\n')
        relevant_departments = list(dep_chosen_result.keys())
        save_to_md(file_name, f"\n------相关科室------\n{', '.join(relevant_departments)}\n-------------------\n")

        relevant_agents = self.initialize_medical_agents(relevant_departments)
        responses = self.collect_responses(relevant_agents, query)

        knowledge_subgraphs = generate_subgraphs(query, self.graph, self.device)
        combined_responses = self.combine_responses_with_knowledge(responses)

        final_response = self.summarize_with_leader_agent(knowledge_subgraphs,combined_responses)
        save_to_md(file_name, f"\n---------------\n最终问诊结果:\n{final_response}\n---------------\n")

        ori_model_output = get_original_model_output(query,self.model, self.tokenizer, self.device)

        return final_response, ori_model_output

    def predict_departments(self, query):
        return predict_department(query,self.departments, self.device)

    def initialize_medical_agents(self, relevant_departments):
        return [MedicalAgent(dep, self.model, self.tokenizer, self.device) for dep in relevant_departments]

    def collect_responses(self, agents, query):
        responses = {}
        for agent in agents:
            response = agent.generate_response(query)
            responses[agent.department_name] = response
        return responses

    def combine_responses_with_knowledge(self, responses):
        combined = ""
        for department, response in responses.items():
            combined += f"【{department}的建议】\n{response}\n"
        return combined

    def summarize_with_leader_agent(self,knowledge_subgraphs,combined_responses):
        prompt = (
            f"你是这次会诊的主治医生，需要汇总科室专家的意见并做出最终诊断。"
            f"以下是各科室专家的详细诊断建议。请你基于这些内容进行全面分析，"
            f"结合每个专家的意见，给出综合诊断，并排除你认为不合理的部分。"
            f"同时，建议进一步的检查、潜在的治疗方案，以及你认为需要重点关注的病情发展。"
            f"请在总结时尽量全面、严谨，确保没有遗漏重要信息。\n"
            f"这里是可能相关的知识内容：\n{knowledge_subgraphs}\n"
            f"其他科室专家意见如下：\n{combined_responses}\n"
            f"请你总结他们的发言，做出全面的诊断总结："
        )
        save_to_md(file_name,f"\n\n汇总prompt:\n{prompt}\n\n")
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id)
        final_response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return final_response

# 主流程函数
def run_medical_consultation(query,model,tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):

    graph = connect_to_graph()

    leader_agent = LeaderAgent(model, tokenizer, graph, device)

    start_time = time.time()
    final_answer,ori_model_output = leader_agent.consult(query)
    print(f"最终问诊结果: {final_answer}")
    save_to_md(file_name, f"\n---------------\n最终问诊结果:\n {final_answer}")
    end_time = time.time()
    timeRecord(start_time, end_time)

    return final_answer,ori_model_output