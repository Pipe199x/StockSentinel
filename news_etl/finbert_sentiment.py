"""FinBERT sentiment enrichment (free, local inference).

Replaces the former Azure AI Language dependency. Uses ProsusAI/finbert, a
BERT model fine-tuned on financial text, running on CPU. The model (~440MB)
is downloaded from the Hugging Face Hub on first use and cached under
~/.cache/huggingface, which the GitHub Actions workflow persists between runs.
"""
import logging
from typing import Any, Dict, List

LOG = logging.getLogger(__name__)


class FinBertSentiment:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            # Lazy import so the module (and news_etl) imports fine without torch installed.
            from transformers import pipeline
            LOG.info("[SENTIMENT] Loading %s (device=%s)…", self.model_name, self.device)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1 if self.device == "cpu" else self.device,
                truncation=True,
                max_length=self.max_length,
            )
        return self._pipeline

    def enrich(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Set item["sentiment"] = {"label", "score"} on items that lack it.

        Mutates and returns the same list. Skips gracefully if transformers
        is not installed.
        """
        pending = [it for it in items if not it.get("sentiment")]
        if not pending:
            return items
        try:
            pipe = self._load()
        except Exception as exc:  # ImportError or download failure
            LOG.warning("[SENTIMENT] FinBERT unavailable (%s); skipping enrichment.", exc)
            return items

        texts = [
            " ".join(str(x) for x in [it.get("title") or "", it.get("summary") or ""] if x)[:2000]
            or "(empty)"
            for it in pending
        ]
        results = pipe(texts, batch_size=self.batch_size)
        for it, res in zip(pending, results):
            it["sentiment"] = {
                "label": res["label"].lower(),
                "score": round(float(res["score"]), 4),
            }
            it["sentiment_model"] = self.model_name
        LOG.info("[SENTIMENT] Scored %d new article(s) with FinBERT.", len(pending))
        return items


def finbert_enrich(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Config-driven entry point mirroring the old azure_enrich signature."""
    if not cfg.get("enabled", True) or not items:
        return items
    client = FinBertSentiment(
        model_name=cfg.get("model", "ProsusAI/finbert"),
        device=cfg.get("device", "cpu"),
        batch_size=int(cfg.get("batch_size", 16)),
        max_length=int(cfg.get("max_length", 512)),
    )
    return client.enrich(items)
