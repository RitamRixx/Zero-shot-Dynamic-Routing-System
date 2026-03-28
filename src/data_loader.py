import json
from src.config import DATA_PATH

def load_intents(data_path: str = DATA_PATH) -> list[str]:
    with open(data_path,'r') as f:
        data = json.load(f)

    intents = list({sample[1] for sample in data["train"]})
    return sorted(intents)

def load_splits(data_path: str = DATA_PATH) -> tuple[list, list]:

    with open(data_path, "r") as f:
        data = json.load(f)
 
    return data["train"], data["test"]