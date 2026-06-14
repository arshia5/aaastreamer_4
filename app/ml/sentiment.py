"""Sentiment inference using the fine-tuned DistilBERT model.

5-class (1..5 stars) classifier; score = expected value over the softmax,
rescaled [1,5] -> [1,10] (always within [0, 10]).
"""
from __future__ import annotations

import threading

from app.ml import config


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


class SentimentModel:
    _instance: "SentimentModel | None" = None
    _lock = threading.Lock()

    def __init__(self):
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )

        model_dir = str(config.SENTIMENT_MODEL_DIR)
        self._torch = torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_dir
        ).to(self.device)
        self.model.eval()
        self.class_scores = torch.tensor(config.SENTIMENT_CLASS_SCORES)

    @classmethod
    def instance(cls) -> "SentimentModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, texts: list[str]) -> list[float]:
        """Return 0..10 sentiment scores for a batch of review texts."""
        torch = self._torch
        out: list[float] = []
        bs = config.SENTIMENT_BATCH_SIZE
        clean = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        for i in range(0, len(clean), bs):
            batch = clean[i:i + bs]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=config.SENTIMENT_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            scores_1_5 = (probs * self.class_scores).sum(dim=-1)
            scores_1_10 = (scores_1_5 - 1.0) / 4.0 * 9.0 + 1.0
            out.extend(_clamp(float(s)) for s in scores_1_10)
        return out

    def predict_one(self, text: str) -> float:
        return self.predict([text])[0]
