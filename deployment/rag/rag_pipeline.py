"""RAG pipeline: vector retrieval + LLM reasoning for low-confidence predictions.

LLM calls go through OpenRouter (OpenAI-compatible API), so any model OpenRouter
hosts can be used by changing `model`. Requires OPENROUTER_API_KEY in the
environment.
"""
import base64
import io
import json
import logging
import os
import time

import chromadb
import requests
import yaml
from chromadb.utils import embedding_functions

from deployment.rag.data_sources import load_knowledge_documents
from deployment.rag.retrievers import scholar

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RAG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "rag_config.yaml")
_INFERENCE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "inference_config.yaml")


class PlantDiseaseRAG:
    def __init__(self, persist_directory=None, model=None):
        inference_cfg = yaml.safe_load(open(_INFERENCE_CONFIG_PATH))["rag"]["llm"]

        self.api_key = os.environ.get(inference_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        if not self.api_key:
            raise RuntimeError(f"{inference_cfg.get('api_key_env', 'OPENROUTER_API_KEY')} is not set")

        cfg = yaml.safe_load(open(_RAG_CONFIG_PATH))
        self.top_k = cfg["retrieval"]["top_k"]
        self.prompt_template = cfg["prompts"]["disease_reasoning"]
        self.identification_prompt = cfg["prompts"]["unknown_plant_identification"]
        self.unknown_plant_prompt = cfg["prompts"]["unknown_plant_reasoning"]
        self.model = model or inference_cfg["model"]

        persist_directory = persist_directory or os.path.join(os.path.dirname(__file__), "vector_store")
        self.client = chromadb.PersistentClient(path=persist_directory)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="plant_diseases", embedding_function=embedding_fn
        )
        if self.collection.count() == 0:
            self._seed()

    def _seed(self):
        docs = load_knowledge_documents()
        self.collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
        )

    def search(self, query, top_k=None):
        results = self.collection.query(query_texts=[query], n_results=top_k or self.top_k)
        docs = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            docs.append({"text": doc, "source": meta.get("source", "unknown"), "distance": dist})
        return docs

    def query(self, crop, disease, confidence):
        start = time.time()
        docs = self.search(f"{crop} {disease} symptoms treatment")

        try:
            docs += scholar.query_disease_literature(crop, disease, retmax=2)
        except requests.RequestException as e:
            logger.warning("PubMed lookup failed (%s) — continuing with vector-store docs only.", e)

        prompt = self.prompt_template.format(
            crop=crop,
            disease=disease,
            confidence=round(confidence * 100, 1),
            docs="\n\n".join(f"[{d['source']}] {d['text']}" for d in docs),
        )

        reasoning = self._chat(prompt)

        return {
            "primary": disease,
            "reasoning": reasoning,
            "sources": sorted({d["source"] for d in docs}),
            "latency_seconds": round(time.time() - start, 2),
        }

    def _chat(self, content):
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [{"role": "user", "content": content}]},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _vision_identify(self, image):
        """Ask a vision-capable LLM to name the plant + disease in a photo.

        Used when the trained classifiers don't cover the crop, so this
        project isn't limited to Tomato/Potato/Pepper.
        """
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        raw = self._chat([
            {"type": "text", "text": self.identification_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])

        # Models sometimes wrap JSON in prose or a code fence despite instructions.
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Vision model did not return JSON: {raw!r}")
        return json.loads(raw[start:end + 1])

    def diagnose_unknown_plant(self, image):
        """Identify + diagnose a plant outside the trained crop set.

        Vision LLM identifies the plant/disease from the photo, then the
        same retrieval (vector store + PubMed) used for known crops grounds
        a second LLM call, so the answer isn't just an ungrounded guess.
        """
        start = time.time()
        identification = self._vision_identify(image)
        plant = identification.get("plant", "unknown plant")
        disease = identification.get("disease", "unknown")
        is_healthy = bool(identification.get("is_healthy", False))
        vision_confidence = identification.get("confidence")

        docs = []
        if not is_healthy:
            docs = self.search(f"{plant} {disease} symptoms treatment")
            try:
                docs += scholar.query_disease_literature(plant, disease, retmax=2)
            except requests.RequestException as e:
                logger.warning("PubMed lookup failed (%s) — continuing with vector-store docs only.", e)

        prompt = self.unknown_plant_prompt.format(
            plant=plant,
            disease=disease,
            healthy_note="" if not is_healthy else " (appears healthy)",
            docs="\n\n".join(f"[{d['source']}] {d['text']}" for d in docs) or "No relevant documents retrieved.",
        )
        reasoning = self._chat(prompt)

        return {
            "plant": plant,
            "disease": disease,
            "is_healthy": is_healthy,
            "vision_confidence": vision_confidence,
            "reasoning": reasoning,
            "sources": sorted({d["source"] for d in docs}),
            "latency_seconds": round(time.time() - start, 2),
        }
