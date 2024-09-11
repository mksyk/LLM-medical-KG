import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cmner import *
from py2neo import Graph
import time

def save_to_md(file_name, content):
    with open(file_name, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

file_name = "test_outputs.md"

profile = "bolt://neo4j:Crj123456@localhost:7687"
graph = Graph(profile)

set_llm = 'llama'
if set_llm == 'llama':
    model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
elif set_llm =='Qwen':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

departments = ['内科', '外科', '五官科', '皮肤性病科', '儿科', '妇产科', '肿瘤科', '传染科','中医科','急诊科','精神科','营养科','心理科','男科','其他科室']

# 科室智能体定义
class MedicalAgent:
    def __init__(self, department_name, model, tokenizer,device):
        self.department_name = department_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    #应该同时接受其对应的1跳子图
    #然后扩展子图数量
    #然后对子图pruning
    #此时pruning方法，参考一跳，2跳等的论文，不对，保持原来的方法
    def generate_response(self, query):
        # 构造适合该科室的prompt，并生成科室的回答
        prompt = f"""你是一名经验丰富的{self.department_name}专家，请根据以下患者信息提供专业意见：{query}\n你的发言："""
        
        #inputs的修改思路：基于论文Knowledge Prompting in Pre-trained Language Model for Natural Language Understanding修改
        #改为基于思维导图的形式
        #思维导图的验证方式，算法参考Boosting Language Models Reasoning with Chain-of-Knowledge Prompting的F2
        #将query和对应思维导图堆叠
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens = 500, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        save_to_md(file_name,f"-------{self.department_name}专家发言--------\n"+response)

        return response
    
    def infer_mindmap(self,subgraphs):
        #example 编写
        example = """"""
        triple_in_text_form = triple_to_text_triple(subgraphs)
        prompt = f""""""
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens = 500, pad_token_id=self.tokenizer.eos_token_id)
        mind_map = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return mind_map

    def evaluate_truth_of_mind_map(self,mind_map,query):
        #判断mind_map的实体是否都来自于知识图谱
        #判断mind_map转成自然语言后，其与query的匹配程度，即truth
        pass
        


class LeaderAgent:
    def __init__(self, model, tokenizer, agents, graph,device):
        self.model = model  
        self.tokenizer = tokenizer
        self.agents = agents  
        self.graph = graph 
        self.device = device

    def consult(self, query):
        save_to_md(file_name,query)
        entities = extract_entities(query)
        print("__________提取实体___________")
        print(entities)
        print("____________________________")
        relevant_agents = self.decide_agents_via_leader(entities,query)
        knowledge_subgraphs = generate_subgraphs(query,graph, model, tokenizer,device,True)#重复提取实体，可以优化
        #疑问：leader 期望根据获取的实体，将对应的子图交给对应的科室。特殊情况：其中某些子图不分配到某些科室，原因：科室不够充分、产生的子图为冗余子图，需要pruning(这个在上一行实现了)
        responses = self.collect_responses(relevant_agents, query)
        combined_responses = self.combine_responses_with_knowledge(responses, knowledge_subgraphs)#需要通过加权投票机制，保留并融合相关科室智能体给出的答案，各科室对于不明确的信息，给出显式的反应
        final_response = self.summarize_with_leader_agent(combined_responses)#格式化输出
        return final_response

    #query实际是query_entities
    #examples是否需要修改
    def decide_agents_via_leader(self,entities,query):
        # Few-shot 示例
        few_shot_examples = """
        这是几个输入输出的示例，你的输出应该仿照示例中的输出。
        示例 1:
        输入: 患者的主要症状或诊断实体包括：高血压、糖尿病。请根据这些信息，从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。
        输出: 内科、内分泌科

        示例 2:
        输入: 患者的主要症状或诊断实体包括：急性胃炎、腹痛。请根据这些信息，从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。
        输出: 内科、消化科

        示例 3:
        输入: 患者的主要症状或诊断实体包括：冠心病、胸痛。请根据这些信息，从以下科室中选择参与会诊的科室：内科、外科、五官科、皮肤性病科、儿科、妇产科、肿瘤科、其他科室。
        输出: 内科、心血管科

        请根据以下患者的主要症状或诊断选择参与会诊的科室。
        """

        prompt = f"患者的病情描述：{query}。请从以下科室中选择参与会诊的科室：{','.join(departments)}。你选择的科室："
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        output = self.model.generate(
            max_length =  len(prompt) + 25,
            **inputs,
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        extracted_departments = []
        for dept in departments:
            if dept in response:
                extracted_departments.append(dept)
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
        prompt = f"以下是各科室医生的会诊结果：\n{combined_responses}\n请你评判它们的答案，生成最终诊断总结。你的总结："

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens = 1024, pad_token_id=self.tokenizer.eos_token_id)
        final_response = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return final_response



#疑问，是否需要对每个科室给出一个单独的模型model，然后微调
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
print(f"运行时长:{end_time - start_time}") 