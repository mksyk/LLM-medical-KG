import json
from collections import Counter

def count_categories(file_path):
    """
    统计JSON文件中 `cate1` 和 `cate2` 的内容及其出现的次数。
    
    参数:
    - file_path: JSON 文件的路径
    """
    # 打开并读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用 Counter 统计 cate1 和 cate2 的出现次数
    cate1_counter = Counter()
    cate2_counter = Counter()

    for entry in data:
        if 'cate1' in entry:
            cate1_counter[entry['cate1']] += 1
        if 'cate2' in entry:
            cate2_counter[entry['cate2']] += 1

    # 输出统计结果
    print("cate1 出现的次数:")
    departments = []
    for cate, count in cate1_counter.items():
        print(f"{cate}: {count}")
        departments.append(cate)

    

    # 保存为 JSON 文件
    with open('data/departments.json', 'w', encoding='utf-8') as f:
        json.dump({item: 1 for item in departments}, f, ensure_ascii=False, indent=4)

    pass
    

# JSON 文件路径
file_path = 'data/CMtMedQA_mapped.json'

# 统计 cate1 和 cate2 的内容及其出现的次数
count_categories(file_path)



pass