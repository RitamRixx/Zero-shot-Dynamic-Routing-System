import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-mnli")
 
OOS_THRESHOLD = float(os.getenv("OOS_THRESHOLD", "0.3"))
 
DATA_PATH = os.getenv("DATA_PATH", "F:/MyProjects/Zero-shot-Dynamic-Routing-System/data/data_full.json")

