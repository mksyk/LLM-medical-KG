import json
import os
def load_json_files(directory):
    """
    从指定目录加载所有 JSON 文件到字典中。
    """
    json_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                prop_name = filename.replace('dict_node_', '').replace('.json', '')
                json_data[prop_name] = json.load(f)
    return json_data

def get_node_properties_by_name(node_name, directory="data/properties"):
    """
    根据节点名称从 JSON 文件中提取该节点的所有属性。
    """
    # 加载所有JSON文件
    data = load_json_files(directory)
    
    # 查找节点名称对应的ID
    node_id = None
    for node_key, node_value in data['name'].items():
        if node_value == node_name:
            node_id = node_key
            break
    
    if node_id is None:
        return f"节点名称'{node_name}'不存在。"

    # 提取该节点的所有属性
    node_properties = {}
    for prop_name, prop_data in data.items():
        node_properties[prop_name] = prop_data.get(node_id, "")

    return node_properties


def generate_node_description(properties):
    """
    根据节点的属性生成描述文本。
    """

    description = []

    # 添加疾病名称
    if properties.get('name'):
        description.append(f"疾病名称：{properties['name']}。")

    # 添加描述
    if properties.get('desc'):
        description.append(f"疾病描述：{properties['desc']}")

    # 添加预防信息
    if properties.get('prevent'):
        description.append(f"预防措施：{properties['prevent']}")

    # 添加治疗方法
    if properties.get('cure_way'):
        cure_ways = "、".join(properties['cure_way']) if isinstance(properties['cure_way'], list) else properties['cure_way']
        description.append(f"治疗方法：{cure_ways}")

    # 添加治疗时长
    if properties.get('cure_lasttime'):
        description.append(f"治疗时长：{properties['cure_lasttime']}")

    # 添加治愈概率
    if properties.get('cured_prob'):
        description.append(f"治愈概率：{properties['cured_prob']}")

    # 添加发病原因
    if properties.get('cause'):
        description.append(f"发病原因：{properties['cause']}")

    # 添加易患人群
    if properties.get('easy_get'):
        description.append(f"易患人群：{properties['easy_get']}")

    # 添加就诊科室
    if properties.get('cure_department'):
        cure_departments = "、".join(properties['cure_department']) if isinstance(properties['cure_department'], list) else properties['cure_department']
        description.append(f"推荐就诊科室：{cure_departments}")

    # 将各段描述合并为最终文本
    return "\n".join(description)


node_name = "肺泡蛋白质沉积症"
properties = get_node_properties_by_name(node_name)
print(properties)
description_text = generate_node_description(properties)
print(description_text)