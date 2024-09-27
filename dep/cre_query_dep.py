


import json
from dep.get_departments import get_dep


def extract_and_save_data(input_file, output_file):
    """
    从指定的 JSON 文件中提取 query 和 label，并保存为新的 JSON 文件。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    departments,dep_size = get_dep()
    result = []
    for entry in data:
        # 如果有 history，针对每个列表的第一个字符串生成单独的数据
        if entry["history"]:
            for history_item in entry["history"]:
                query = history_item[0]  # 获取每个 history 列表中的第一个字符串
                label1 = entry["cate1"]
                label1 = label1 if label1 in departments else "其他科室"
                
                extracted_data = {
                    "query": query,  # 当前的 query
                    "label": label1,  # 对应的科室
                }
                result.append(extracted_data)
        else:
            # 如果没有 history，使用 instruction 生成一条数据
            query = entry["instruction"]
            label1 = entry["cate1"]
            label1 = label1 if label1 in departments else "其他科室"
            
            extracted_data = {
                "query": query,  # 使用 instruction 作为 query
                "label": label1,  # 对应的科室
            }
            result.append(extracted_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {output_file}")