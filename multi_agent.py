import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cmner import *
from py2neo import Graph

#连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)


# 加载模型与tokenizer
model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 科室智能体定义
class MedicalAgent:
    def __init__(self, department_name, model, tokenizer):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, query):
        # 构造适合该科室的prompt，并生成科室的回答
        prompt = f"你是一名经验丰富的{self.department_name}专家，请根据以下患者信息提供专业意见：" \
                 f"\n{query}" \
                 "\n请直接回答可能的诊断、检查、治疗方案和注意事项，避免重复症状描述。"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_length=512, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.strip()



class LeaderAgent:
    def __init__(self, leader_agent_model, tokenizer, agents, graph,device):
        self.leader_agent_model = leader_agent_model  # 大模型，用于领导agent决策
        self.tokenizer = tokenizer
        self.agents = agents  # 包含所有科室的MedicalAgents
        self.graph = graph  # 知识图谱
        self.device = device

    def consult(self, query):
        # 1. 提取实体
        entities = extract_entities(query)

        # 2. 让leader agent决定需要参与的科室
        relevant_agents = self.decide_agents_via_leader(entities)

        # 3. 根据 query 和图谱生成子图知识
        knowledge_subgraphs = generate_subgraphs(query)

        # 4. 收集各科室的回答
        responses = self.collect_responses(relevant_agents, query)

        # 5. 整合子图知识和科室回答
        combined_responses = self.combine_responses_with_knowledge(responses, knowledge_subgraphs)

        # 6. 最后生成leader agent的总结
        final_response = self.summarize_with_leader_agent(combined_responses)
        return final_response

    def decide_agents_via_leader(self, entities):
        # 生成 prompt 让 leader agent 决定需要哪些科室
        prompt = f"患者的主要症状或诊断实体包括：{', '.join(entities)}。请根据这些信息，从以下科室中选择参与会诊的科室：" \
                 f"内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。请列出需要的科室。"

        # 通过大模型推理得到需要的科室
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.leader_agent_model.generate(**inputs, max_length=512, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 从返回结果中提取科室
        relevant_departments = self.parse_leader_response(response)
        
        # 根据科室选择对应的 agents
        return [agent for agent in self.agents if agent.department_name in relevant_departments]

    def parse_leader_response(self, response):
        # 假设模型返回格式为 "需要的科室有：内科、外科"
        print(response)
        return [dept.strip() for dept in response.split('有：')[1].split('、')]

    def collect_responses(self, agents, query):
        responses = {}
        for agent in agents:
            response = agent.generate_response(query)
            responses[agent.department_name] = response
            print(f"{agent.department_name} 的回答: {response}")
        return responses

    def combine_responses_with_knowledge(self, responses, knowledge_subgraphs):
        # 将各科室的回答和知识子图整合（简单拼接示例，可根据需要修改）
        combined = ""
        for department, response in responses.items():
            combined += f"【{department}的建议】\n{response}\n"
        combined += "\n【相关的知识图谱内容】\n" + "\n".join(knowledge_subgraphs)
        return combined

    def summarize_with_leader_agent(self, combined_responses):
        # 构造prompt让leader agent生成最终总结
        prompt = f"以下是各科室医生的会诊结果和相关知识：\n{combined_responses}\n请根据这些内容生成最终诊断总结。"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        output = self.leader_agent_model.generate(**inputs, max_length=512, pad_token_id=self.tokenizer.eos_token_id)
        final_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return final_response.strip()




# 科室智能体的初始化
departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '其他科室']
agents = {dep: MedicalAgent(dep, model, tokenizer) for dep in departments}

# leader agent的初始化
leader_agent = LeaderAgent( model, tokenizer, agents,graph,device)

# 输入问诊 query
query = """患者既往慢阻肺多年;
冠心病史6年，平素规律服用心可舒、保心丸等控制可;双下肢静脉血栓3年，保守治疗效果可;
左侧腹股沟斜疝无张力修补术后2年。
否认"高血压、糖尿病"等慢性病病史，否认"肝炎、结核"等传染病病史及其密切接触史，
否认其他手术、重大外伤、输血史，否认"食物、药物、其他"等过敏史，预防接种史随社会。
"""
final_answer = leader_agent.consult(query)
print(f"最终问诊结果: {final_answer}")