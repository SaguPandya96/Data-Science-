"""Model wrapper.

`unitary/toxic-bert` is multi-label (toxic, severe_toxic, obscene, threat, insult,
identity_hate). Everything downstream wants one number, so this collapses it to the
`toxic` head's probability.
"""

from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_ID = "unitary/toxic-bert"


class Scorer:
    """Batched toxicity scoring over a list of strings."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = None,
        batch_size: int = 32,
        max_length: int = 256,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()

        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.toxic_idx = labels.get("toxic", 0)

    @torch.no_grad()
    def __call__(self, texts: List[str]) -> np.ndarray:
        """Return P(toxic) for each input."""
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.sigmoid(self.model(**enc).logits)[:, self.toxic_idx]
            scores.extend(probs.cpu().numpy().tolist())
        return np.array(scores)
