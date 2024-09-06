import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cmner import *
from py2neo import Graph

def save_to_md(file_name, content):
    with open(file_name, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

file_name = "test_outputs.md"

# 科室智能体定义
class MedicalAgent:
    def __init__(self, department_name, model, tokenizer,device):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, query):
        # 构造适合该科室的prompt，并生成科室的回答
        prompt = f"你是一名经验丰富的{self.department_name}专家，请根据以下患者信息提供专业意见：" \
                 f"\n{query}" \
                 "\n请直接回答可能的诊断、检查、治疗方案和注意事项，避免重复症状描述。"
        
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_length=4096, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        print(f"-------{self.department_name}专家发言--------")
        save_to_md(file_name,f"-------{self.department_name}专家发言--------\n"+response)

        return response



class LeaderAgent:
    def __init__(self, model, tokenizer, agents, graph,device):
        self.model = model  # 大模型，用于领导agent决策
        self.tokenizer = tokenizer
        self.agents = agents  # 包含所有科室的MedicalAgents
        self.graph = graph  # 知识图谱
        self.device = device

    def consult(self, query):
        # 1. 提取实体
        entities = extract_entities(query)
        print("__________提取实体___________")
        print(entities)
        # 2. 让leader agent决定需要参与的科室
        relevant_agents = self.decide_agents_via_leader(entities)

        # 3. 根据 query 和图谱生成子图知识
        knowledge_subgraphs = generate_subgraphs(query,graph, model, tokenizer,device,True)

        # 4. 收集各科室的回答
        responses = self.collect_responses(relevant_agents, query)

        # 5. 整合子图知识和科室回答
        combined_responses = self.combine_responses_with_knowledge(responses, knowledge_subgraphs)

        # 6. 最后生成leader agent的总结
        final_response = self.summarize_with_leader_agent(combined_responses)
        return final_response

    def decide_agents_via_leader(self, entities):
        # Few-shot 示例
        few_shot_examples = """
        示例 1:
        输入: 患者的主要症状或诊断实体包括：高血压、糖尿病。请根据这些信息，从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。
        输出: 内科、内分泌科

        示例 2:
        输入: 患者的主要症状或诊断实体包括：急性胃炎、腹痛。请根据这些信息，从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。
        输出: 内科、消化科

        示例 3:
        输入: 患者的主要症状或诊断实体包括：冠心病、胸痛。请根据这些信息，从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。
        输出: 内科、心血管科

        请根据以下患者的主要症状或诊断实体选择参与会诊的科室：
        """

        prompt = f"{few_shot_examples}\n患者的主要症状或诊断实体包括：{', '.join(entities)}。请从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。你选择的科室："
        # 编码输入文本
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # 通过大模型推理得到需要的科室
        output = self.model.generate(
            max_length =  len(prompt) + 25,
            **inputs,
        )

        # 解码输出
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        extracted_departments = []
        for dept in departments:
            if dept in response:
                extracted_departments.append(dept)
        print("------departments------")
        print(extracted_departments)
        save_to_md(file_name,f"------departments------\n"+ ','.join(extracted_departments) + "\n-----------------------")
        print("-----------------------")
        return [agent for agent in self.agents.values() if agent.department_name in extracted_departments]

    def collect_responses(self, agents, query):
        responses = {}
        for agent in agents:
            response = agent.generate_response(query)
            responses[agent.department_name] = response
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

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_length=8192, pad_token_id=self.tokenizer.eos_token_id)
        final_response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return final_response


#连接到neo4j，获得知识图谱graph
profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)


# 加载模型与tokenizer
model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 科室智能体的初始化
departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科','中医科','急诊科','精神科','营养科','心理科','男科','其他科室']
agents = {dep: MedicalAgent(dep, model, tokenizer,device) for dep in departments}

# leader agent的初始化
leader_agent = LeaderAgent(model, tokenizer, agents,graph,device)

# 输入问诊 query
query = """患者既往慢阻肺多年;
冠心病史6年，平素规律服用心可舒、保心丸等控制可;双下肢静脉血栓3年，保守治疗效果可;
左侧腹股沟斜疝无张力修补术后2年。
否认"高血压、糖尿病"等慢性病病史，否认"肝炎、结核"等传染病病史及其密切接触史，
否认其他手术、重大外伤、输血史，否认"食物、药物、其他"等过敏史，预防接种史随社会。
"""
final_answer = leader_agent.consult(query)
print(f"最终问诊结果: {final_answer}")
save_to_md(file_name,f"---------------\n"+f"最终问诊结果:\n {final_answer}")