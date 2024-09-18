'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-14 09:42:48
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-18 08:14:43
FilePath: /LLM-medical-KG/multi_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from cmner import *
from py2neo import Graph
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

file_name = "test_outputs.md"


profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)


set_llm = 'deepseek'
if set_llm == 'llama':
    model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
elif set_llm =='Qwen':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
elif set_llm == 'deepseek':
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id




device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model.to(device)

departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科','中医科','急诊科','精神科','营养科','心理科','男科','其他科室']

# 科室智能体定义
class MedicalAgent:
    def __init__(self, department_name, model, tokenizer,device):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, query):
        # 构造适合该科室的prompt，并生成科室的回答
        prompt = f"""你是一名经验丰富的{self.department_name}专家，请根据以下患者信息提供专业意见：{query}\n你的发言："""
        print(f"{self.department_name}发言中..." )
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens = 500, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        save_to_md(file_name,f"\n-------{self.department_name}专家发言--------\n"+response)

        return response



class LeaderAgent:
    def __init__(self, model, tokenizer, agents, graph,device):
        self.model = model  
        self.tokenizer = tokenizer
        self.agents = agents  
        self.graph = graph 
        self.device = device

    def consult(self, query):
        save_to_md(file_name,query)
        relevant_agents = self.decide_agents_via_leader(query)
        knowledge_subgraphs = generate_subgraphs(query,graph,device)
        responses = self.collect_responses(relevant_agents, query)
        combined_responses = self.combine_responses_with_knowledge(responses, knowledge_subgraphs)
        final_response = self.summarize_with_leader_agent(combined_responses)
        return final_response

    def decide_agents_via_leader(self, query):
        
        extracted_departments = departments#暂时先不要这个挑的过程了
        save_to_md(file_name,f"------departments------\n"+ ','.join(extracted_departments) + "\n-----------------------")
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
        prompt = f"以下是各科室医生的会诊结果和相关知识：\n{combined_responses}\n请根据这些内容生成最终诊断总结。你的总结："

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens = 1024, pad_token_id=self.tokenizer.eos_token_id)
        final_response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return final_response




agents = {dep: MedicalAgent(dep, model, tokenizer,device) for dep in departments}


leader_agent = LeaderAgent(model, tokenizer, agents,graph,device)


query = """患者既往慢阻肺多年;
冠心病史6年，平素规律服用心可舒、保心丸等控制可;双下肢静脉血栓3年，保守治疗效果可;
左侧腹股沟斜疝无张力修补术后2年。
否认"高血压、糖尿病"等慢性病病史，否认"肝炎、结核"等传染病病史及其密切接触史，
否认其他手术、重大外伤、输血史，否认"食物、药物、其他"等过敏史，预防接种史随社会。
"""
start_time = time.time()
final_answer = leader_agent.consult(query)
print(f"最终问诊结果: {final_answer}")
save_to_md(file_name,f"---------------\n"+f"最终问诊结果:\n {final_answer}")
end_time = time.time()
timeRecord(start_time,end_time)