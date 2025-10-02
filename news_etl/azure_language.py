# news_etl/azure_language.py
import os
import json
import requests
from typing import List, Dict, Any, Optional
from collections import defaultdict

_ALLOWED_KINDS = {
    "LanguageDetection",
    "SentimentAnalysis",
    "KeyPhraseExtraction",
    "EntityRecognition",
    "EntityLinking",
}
_MAX_DOCS_PER_REQUEST = 5


def _chunk(it, size):
    for i in range(0, len(it), size):
        yield it[i : i + size]


class AzureLanguageClient:
    def __init__(
        self,
        endpoint: str,
        key: str,
        api_version: str = "2023-04-01",
        force_language: Optional[str] = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.key = key or os.getenv("AZURE_LANGUAGE_KEY", "")
        if not self.key:
            raise ValueError("Azure Language key no configurada (faltan 'key' o AZURE_LANGUAGE_KEY).")
        self.api_version = api_version
        self.force_language = force_language
        self.timeout_seconds = timeout_seconds

        self.url_single = f"{self.endpoint}/language/:analyze-text"
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_task_list(self, flags: Dict[str, bool]) -> List[str]:
        kinds: List[str] = []
        if flags.get("language_detection"):
            kinds.append("LanguageDetection")
        if flags.get("sentiment_analysis"):
            kinds.append("SentimentAnalysis")
        if flags.get("key_phrase_extraction"):
            kinds.append("KeyPhraseExtraction")
        if flags.get("entity_recognition"):
            kinds.append("EntityRecognition")
        if flags.get("entity_linking"):
            kinds.append("EntityLinking")
        if not kinds:
            raise ValueError("No tasks habilitadas en configuración.")
        for k in kinds:
            if k not in _ALLOWED_KINDS:
                raise ValueError(f"Tarea no soportada por API {self.api_version}: {k}")
        return kinds

    @staticmethod
    def _needs_language_field(kind: str) -> bool:
        return kind != "LanguageDetection"

    def _attach_language_if_needed(self, docs: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        if self.force_language and self._needs_language_field(kind):
            return [{**d, "language": d.get("language", self.force_language)} for d in docs]
        return docs

    def _default_params_for(self, kind: str) -> Dict[str, Any]:
        p: Dict[str, Any] = {"modelVersion": "latest"}
        if kind == "SentimentAnalysis":
            p["opinionMining"] = False
        return p

    def _call_single_task(self, kind: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "kind": kind,
            "parameters": self._default_params_for(kind),
            "analysisInput": {"documents": documents},
        }
        print(f"[AzureLanguage] POST {kind} payload:", json.dumps(payload, ensure_ascii=False)[:900])

        resp = requests.post(
            self.url_single,
            headers=self.headers,
            params={"api-version": self.api_version},
            json=payload,
            timeout=self.timeout_seconds,
        )
        if not resp.ok:
            print("=== Azure error body ===")
            print(f"HTTP {resp.status_code} {resp.reason}")
            try:
                print(resp.json())
            except Exception:
                print(resp.text)
            resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _inner_results(json_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        El endpoint single-task suele devolver:
          {"kind":"...Results","results":{ "documents":[...] , "errors":[...], ...}}
        Devolvemos SIEMPRE el bloque interno con 'documents'.
        """
        if isinstance(json_obj, dict) and "results" in json_obj and isinstance(json_obj["results"], dict):
            return json_obj["results"]
        return json_obj  # fallback si ya viene "plano"

    @staticmethod
    def _wrap_grouped(kind: str, result_json: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            "LanguageDetection": "languageDetectionTasks",
            "SentimentAnalysis": "sentimentAnalysisTasks",
            "KeyPhraseExtraction": "keyPhraseExtractionTasks",
            "EntityRecognition": "entityRecognitionTasks",
            "EntityLinking": "entityLinkingTasks",
        }
        coll = mapping[kind]
        # Guardamos directamente el bloque con 'documents' para que el parser lo lea sin niveles extra
        inner = AzureLanguageClient._inner_results(result_json)
        return {"tasks": {coll: [{"results": inner}]}}

    @staticmethod
    def _merge_grouped(into: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
        if "tasks" not in into:
            into["tasks"] = {}
        for coll, arr in add.get("tasks", {}).items():
            into["tasks"].setdefault(coll, [])
            into["tasks"][coll].extend(arr)
        return into

    def analyze_batch(
        self,
        documents: List[Dict[str, Any]],
        tasks: Optional[Dict[str, bool]] = None,
        task_flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        flags = tasks or task_flags
        if flags is None:
            raise ValueError("Faltan los flags de tareas (tasks/task_flags).")

        kinds = self._build_task_list(flags)

        combined: Dict[str, Any] = {"tasks": {}}
        for kind in kinds:
            for docs_chunk in _chunk(documents, _MAX_DOCS_PER_REQUEST):
                docs_ready = self._attach_language_if_needed(docs_chunk, kind)
                result_json = self._call_single_task(kind, docs_ready)
                grouped = self._wrap_grouped(kind, result_json)
                self._merge_grouped(combined, grouped)

        return combined

    @staticmethod
    def _extract_bucket_results(bucket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Soporta ambos formatos:
          bucket = {"results": {"documents":[...], ...}}
          bucket = {"results": {"results": {"documents":[...], ...}}}
        """
        res = bucket.get("results", {}) or {}
        if isinstance(res, dict) and "documents" in res:
            return res
        if isinstance(res, dict) and "results" in res and isinstance(res["results"], dict):
            return res["results"]
        return {"documents": [], "errors": []}

    @staticmethod
    def parse_results(grouped: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        per_doc = defaultdict(
            lambda: {
                "language": None,
                "sentiment": {"label": None, "score": 0.0},
                "key_phrases": [],
                "entities": [],
                "linked_entities": [],
            }
        )

        tasks = grouped.get("tasks", {})

        # LanguageDetection
        for bucket in tasks.get("languageDetectionTasks", []):
            results = AzureLanguageClient._extract_bucket_results(bucket)
            for d in results.get("documents", []):
                _id = str(d.get("id"))
                lang = d.get("detectedLanguage") or {}
                iso = lang.get("iso6391Name")
                name = lang.get("name")
                score = lang.get("confidenceScore")
                if iso or name:
                    per_doc[_id]["language"] = {
                        "iso": iso,
                        "name": name,
                        "score": float(score) if isinstance(score, (int, float)) else None,
                    }

        # SentimentAnalysis
        for bucket in tasks.get("sentimentAnalysisTasks", []):
            results = AzureLanguageClient._extract_bucket_results(bucket)
            for d in results.get("documents", []):
                _id = str(d.get("id"))
                label = d.get("sentiment")
                scores = d.get("confidenceScores") or {}
                sel_score = None
                if isinstance(scores, dict) and label in scores:
                    try:
                        sel_score = float(scores.get(label))
                    except Exception:
                        sel_score = None
                if sel_score is None and isinstance(scores, dict):
                    try:
                        sel_score = float(max(scores.values()))
                    except Exception:
                        sel_score = 0.0
                per_doc[_id]["sentiment"] = {"label": label, "score": sel_score or 0.0}

        # KeyPhraseExtraction
        for bucket in tasks.get("keyPhraseExtractionTasks", []):
            results = AzureLanguageClient._extract_bucket_results(bucket)
            for d in results.get("documents", []):
                _id = str(d.get("id"))
                kps = d.get("keyPhrases") or []
                seen = set()
                clean = []
                for k in kps:
                    k = (k or "").strip()
                    if k and k.lower() not in seen:
                        seen.add(k.lower())
                        clean.append(k)
                per_doc[_id]["key_phrases"] = clean

        # EntityRecognition
        for bucket in tasks.get("entityRecognitionTasks", []):
            results = AzureLanguageClient._extract_bucket_results(bucket)
            for d in results.get("documents", []):
                _id = str(d.get("id"))
                ents = d.get("entities") or []
                names: List[str] = []
                seen = set()
                for e in ents:
                    text = (e.get("text") or "").strip()
                    if text and text.lower() not in seen:
                        seen.add(text.lower())
                        names.append(text)
                per_doc[_id]["entities"] = names

        # EntityLinking
        for bucket in tasks.get("entityLinkingTasks", []):
            results = AzureLanguageClient._extract_bucket_results(bucket)
            for d in results.get("documents", []):
                _id = str(d.get("id"))
                ents = d.get("entities") or []
                names: List[str] = []
                seen = set()
                for e in ents:
                    name = (e.get("name") or "").strip()
                    if name and name.lower() not in seen:
                        seen.add(name.lower())
                        names.append(name)
                per_doc[_id]["linked_entities"] = names

        return per_doc
