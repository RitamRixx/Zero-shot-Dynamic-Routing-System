from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class TFIDFBaseline:
    def __init__(self):
        self._model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self._is_trained = False

    def train(self, utterances: list[str], intents: list[str]) -> None:
        self._model.fit(utterances, intents)
        self._is_trained = True
 
    def predict(self, utterances: list[str]) -> list[str]:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self._model.predict(utterances).tolist()
 
    def predict_proba(self, utterances: list[str]):
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self._model.predict_proba(utterances)