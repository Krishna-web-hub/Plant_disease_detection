"""PubMed retriever via NCBI E-utilities.

Public API, no key required for light use (NCBI asks for a descriptive
User-Agent and recommends staying under ~3 requests/sec without an API key).
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "plant-disease-detection-rag/1.0 (research project; contact via project owner)"


def search(query, retmax=3, timeout=15):
    """Search PubMed, return a list of PMIDs."""
    resp = requests.get(
        f"{EUTILS_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"].get("idlist", [])


def fetch_abstracts(pmids, timeout=15):
    """Fetch abstract text for a list of PMIDs, keyed by PMID."""
    if not pmids:
        return {}

    resp = requests.get(
        f"{EUTILS_BASE}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "text"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()

    records = [r.strip() for r in resp.text.split("\n\n\n") if r.strip()]
    # efetch returns records in the same order as the requested PMIDs.
    return dict(zip(pmids, records))


def query_disease_literature(crop, disease, retmax=3):
    """Search + fetch abstracts for a crop/disease pair. Used by the RAG pipeline."""
    pmids = search(f"{crop} {disease} disease management", retmax=retmax)
    abstracts = fetch_abstracts(pmids)

    docs = []
    for pmid, text in abstracts.items():
        docs.append({
            "text": text[:3000],
            "source": "PubMed",
            "metadata": {"pmid": pmid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"},
        })
    return docs
