from transformers import pipeline
from src.config import MODEL_NAME, OOS_THRESHOLD
from src.data_loader import load_intents

def humanize(label: str) -> str:
    return label.replace("_", " ")

class zeroshotClassifier:
    def __init__(self):
        print(f"[Classifier] Loading model: {MODEL_NAME}")
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
        )
        print("[Classifier] Model loaded.")

        self._labels: list[str] = load_intents()
        print(f"[Classifier] Default labels loaded: {len(self._labels)} intents")

    
    def classify(self, text: str) -> dict:
        human_labels = [humanize(label) for label in self._labels]
        reverse_map = {humanize(l): l for l in self._labels}
 
        result = self._pipeline(text, candidate_labels=human_labels)
 
        top_human_label = result["labels"][0]
        top_score = result["scores"][0]

        return {
            "text": text,
            "intent": reverse_map[top_human_label],
            "confidence": round(top_score, 4),
            "is_oos": top_score < OOS_THRESHOLD,
        }
    

    def get_labels(self) -> list[str]:
        return self._labels.copy()
    
    def update_labels(self, new_labels: list[str]) -> dict:
        old_count = len(self._labels)
        self._labels = new_labels
        new_count = len(self._labels)
 
        print(f"[Classifier] Labels updated: {old_count} -> {new_count}")
        return {
            "previous_count": old_count,
            "new_count": new_count,
            "labels": self._labels,
        }
    
classifier = zeroshotClassifier()