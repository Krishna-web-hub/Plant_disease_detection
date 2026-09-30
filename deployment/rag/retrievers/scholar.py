"""PubMed retriever via NCBI E-utilities with SSRF & Input Sanitization.

Enforces:
1. Strict domain whitelisting (https://eutils.ncbi.nlm.nih.gov only).
2. Strict PMID digit validation to prevent parameter injection.
3. Response payload size limits to guard against memory exhaustion.
4. Input query sanitization.
"""
import logging
import re
import urllib.parse
from typing import Dict, List
import requests

logger = logging.getLogger(__name__)

ALLOWED_DOMAIN = "eutils.ncbi.nlm.nih.gov"
EUTILS_BASE = f"https://{ALLOWED_DOMAIN}/entrez/eutils"
USER_AGENT = "plant-disease-detection-rag/1.0 (academic research; contact via project owner)"
MAX_RESPONSE_BYTES = 512 * 1024  # 512 KB maximum text


def _validate_pmid(pmid: str) -> bool:
    """Verifies PMID consists strictly of numeric digits."""
    return bool(re.match(r"^\d{1,10}$", str(pmid).strip()))


def search(query: str, retmax: int = 3, timeout: int = 10) -> List[str]:
    """Search PubMed safely, returning a validated list of PMIDs."""
    # Sanitize query: alphanumeric, basic punctuation only
    safe_query = re.sub(r"[^\w\s\-\.\,\(\)]", "", query).strip()[:200]
    if not safe_query:
        return []

    try:
        resp = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": safe_query, "retmode": "json", "retmax": min(retmax, 5)},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()

        # Enforce size limit
        if len(resp.content) > MAX_RESPONSE_BYTES:
            logger.warning("PubMed search response exceeded byte limit; truncated.")
            return []

        data = resp.json()
        raw_ids = data.get("esearchresult", {}).get("idlist", [])
        return [pmid for pmid in raw_ids if _validate_pmid(pmid)]
    except Exception as e:
        logger.warning("PubMed search failed: %s", e)
        return []


def fetch_abstracts(pmids: List[str], timeout: int = 10) -> Dict[str, str]:
    """Fetch abstract text for a list of validated PMIDs."""
    valid_pmids = [p for p in pmids if _validate_pmid(p)]
    if not valid_pmids:
        return {}

    try:
        resp = requests.get(
            f"{EUTILS_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(valid_pmids[:5]), "rettype": "abstract", "retmode": "text"},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()

        if len(resp.content) > MAX_RESPONSE_BYTES:
            logger.warning("PubMed efetch response exceeded byte limit.")
            return {}

        records = [r.strip() for r in resp.text.split("\n\n\n") if r.strip()]
        return dict(zip(valid_pmids, records[:len(valid_pmids)]))
    except Exception as e:
        logger.warning("PubMed efetch failed: %s", e)
        return {}


def query_disease_literature(crop: str, disease: str, retmax: int = 3) -> List[Dict]:
    """Search + fetch abstracts for a crop/disease pair with safe query formatting."""
    safe_crop = re.sub(r"[^\w\s]", "", crop).strip()
    safe_disease = re.sub(r"[^\w\s]", "", disease).strip()
    query = f"{safe_crop} {safe_disease} disease management"

    pmids = search(query, retmax=retmax)
    abstracts = fetch_abstracts(pmids)

    docs = []
    for pmid, text in abstracts.items():
        clean_text = text[:3000].replace("\r", " ").strip()
        docs.append({
            "text": clean_text,
            "source": "PubMed",
            "metadata": {"pmid": pmid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"},
        })
    return docs
