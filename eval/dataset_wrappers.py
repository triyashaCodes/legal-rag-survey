# Dataset wrappers module
import json

def load_CUAD(path):
    with open(path) as f:
        data = json.load(f)
    return [
        {"question": item["question"], "answer": item["gold_answer"]}
        for item in data
    ]

