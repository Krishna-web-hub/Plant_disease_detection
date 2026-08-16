"""Registry of RAG data sources, and the document loader that seeds the vector store."""
import json
import os

_TREATMENT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "treatment_db.json")
_EXTENSION_DOCS_PATH = os.path.join(os.path.dirname(__file__), "extension_docs.json")

RAG_SOURCES = {
    "primary": [
        {"id": "plantvillage_internal", "name": "PlantVillage (internal treatment DB)", "type": "knowledge_base", "status": "connected"},
        {"id": "extension_factsheets", "name": "Land-grant Cooperative Extension fact sheets", "type": "official_guidance", "status": "connected",
         "note": "No official USDA disease API exists; these are the real authoritative sources USDA guidance mirrors. See retrievers/usda.py."},
    ],
    "secondary": [
        {"id": "pubmed_central", "name": "PubMed (NCBI E-utilities)", "type": "research_papers", "status": "connected",
         "note": "Queried live at RAG query time, not pre-seeded. See retrievers/scholar.py."},
    ],
}


def load_knowledge_documents():
    """Build the seed corpus for the vector store: internal treatment DB entries
    plus cached extension fact sheets (if the cache has been built -- see
    build_extension_cache.py). PubMed is queried live at query time instead,
    since it's a real searchable API rather than a static corpus.
    """
    with open(_TREATMENT_DB_PATH) as f:
        db = json.load(f)

    docs = []
    for category, entry in db.items():
        if entry["severity"] == "none":
            continue  # "healthy" entries have nothing to retrieve

        text = (
            f"{entry['crop']} - {entry['disease']} ({entry['pathogen_type']}: {entry.get('pathogen') or 'n/a'}).\n"
            f"Symptoms: {'; '.join(entry['symptoms'])}\n"
            f"Treatment: {'; '.join(entry['treatment'])}\n"
            f"Prevention: {'; '.join(entry['prevention'])}"
        )
        docs.append({
            "id": category,
            "text": text,
            "metadata": {"source": "PlantVillage", "crop": entry["crop"], "category": category},
        })

    if os.path.isfile(_EXTENSION_DOCS_PATH):
        with open(_EXTENSION_DOCS_PATH) as f:
            docs.extend(json.load(f))

    return docs
