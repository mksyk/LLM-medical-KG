import pandas as pd
from cmner import *

data_path = "Chinese-medical-dialogue-data/样例_内科5000-6000.csv"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name_or_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3.1-8B-Chinese-Chat/snapshots/404a735a6205e5ef992f589b6d5d28922822928e"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def process_csv_ner(data_path, output_txt):
    df = pd.read_csv(data_path, encoding='gb18030',nrows = 1)
    nodes = []
    # 打开输出文本文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        # 遍历DataFrame的每一行
        for _, row in df.iterrows():
            value = row['ask']  # 获取'ask'列的值
            entities = extract_entities(value)
            f.write(value + '\n')
            nodes.append(entities)
            # 写入实体列表
            f.write(', '.join(entities) + '\n')
    return nodes
process_csv_ner(data_path, 'result_ner.txt')
