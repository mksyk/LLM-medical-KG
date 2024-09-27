import json

def get_dep(file_path = "data/departments.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        departments = list(data.keys())
        dep_size = list(data.values())
    return departments,dep_size

