import json

label_mapping = {
    # 直接映射
    "男科": "男科",
    "肿瘤科": "肿瘤科",
    "中医科": "中医科",
    "整形美容": "整形美容科",
    "妇产科": "妇科",  # 知识图谱中有 "妇科"，但没有明确的 "妇产科"，可以合并
    "儿科": "儿科",  # 知识图谱有"儿科"、"小儿内科"和"小儿外科"，可以根据需求合并或细分
    "外科": "外科",  # 知识图谱有细分外科，可以根据具体需求决定细化
    "五官科": "五官科",  # 知识图谱中有 "耳鼻喉科"，可以合并
    "内科": "内科",
    "传染病科": "传染科",  # 对应知识图谱中的 "传染科"
    "皮肤性病科": "皮肤性病科",  
    "心理科": "心理科",  # 知识图谱有心理科
    "皮肤科": "皮肤科",  # 知识图谱有皮肤科
    "精神科": "精神科",  # 知识图谱有精神科
    "传染科": "传染科",  
    "家居健康": "营养科",  
    "其他": "其他科室",  
    "nan": "其他科室" 
}

# 读取 json 文件
input_file = "data/CMtMedQA.json"
output_file = "data/CMtMedQA_mapped.json"

def map_labels(data, mapping):
    """
    使用映射表将数据中的 label 字段映射为知识图谱中的科室
    """
    for entry in data:
        cate1 = entry.get("cate1", "")
        if cate1 in mapping:
            entry["cate1"] = mapping[cate1]  # 将 cate1 替换为映射后的标签
        else:
            entry["cate1"] = "其他科室"  # 如果没有映射，设为 "其他科室"

        
    return data

# 读取文件，进行映射
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 执行映射
mapped_data = map_labels(data, label_mapping)

# 保存映射后的数据到新的 json 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(mapped_data, f, ensure_ascii=False, indent=4)

print(f"映射完成，输出文件保存为 {output_file}")