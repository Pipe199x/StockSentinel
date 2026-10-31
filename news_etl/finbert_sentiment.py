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


def article_id(item: Dict[str, Any]) -> str:
    """Id estable por artículo (URL, o título + fecha si no hay URL)."""
    import hashlib
    key = item.get("url") or f"{item.get('title', '')}|{item.get('published_at', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def update_embeddings(items: List[Dict[str, Any]], path: str, cfg: Dict[str, Any]) -> int:
    """Guarda el embedding FinBERT (768 dims, media de la última capa) de cada artículo.

    Es la rama de "experto en noticias" del modelo de fusión (stocks_etl/deep): el
    sentimiento solo es una etiqueta, el embedding conserva la semántica. Solo calcula
    los artículos que faltan, así que la primera corrida rellena todo el histórico.
    Devuelve cuántos embeddings nuevos se añadieron.
    """
    import os
    import numpy as np

    if not cfg.get("enabled", True) or not items:
        return 0
    ids, embs = [], np.zeros((0, 768), dtype=np.float16)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            ids, embs = list(z["ids"]), z["emb"]
    known = set(ids)
    pending = [it for it in items if it.get("tickers") and article_id(it) not in known]
    if not pending:
        return 0
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = cfg.get("model", "ProsusAI/finbert")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()
    except Exception as exc:
        LOG.warning("[EMBED] FinBERT unavailable (%s); skipping embeddings.", exc)
        return 0

    texts = [" ".join(str(x) for x in [it.get("title") or "", it.get("summary") or ""] if x)[:2000] or "(empty)"
             for it in pending]
    out = []
    bs = int(cfg.get("batch_size", 16))
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=int(cfg.get("max_length", 512)), return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy().astype(np.float16))
    new = np.concatenate(out)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids + [article_id(it) for it in pending]),
                        emb=np.concatenate([embs, new]))
    LOG.info("[EMBED] Added %d FinBERT embedding(s) to %s.", len(pending), path)
    return len(pending)
