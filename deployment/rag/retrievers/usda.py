"""Official/extension plant-disease guidance retriever.

There is no official USDA REST API for plant disease information: the
third-party "USDA Plants Database API" project is discontinued, and
APHIS's IPHIS system is restricted to state/federal cooperators (not
public). Instead, this fetches real, citable land-grant Cooperative
Extension fact sheets -- the same authoritative sources USDA guidance
typically mirrors -- for a curated set of diseases.

These are static reference pages (not a live query API), so results are
meant to be fetched once via fetch_all() and cached -- see
deployment/rag/build_extension_cache.py.
"""
import re
from html.parser import HTMLParser

import requests

USER_AGENT = "plant-disease-detection-rag/1.0 (research project; contact via project owner)"

# Curated, manually verified extension fact sheets (checked 2026-08).
# Re-verify periodically -- university sites occasionally restructure URLs.
EXTENSION_SOURCES = {
    "Pepper__bell___Bacterial_spot": {
        "url": "https://edis.ifas.ufl.edu/publication/PP362",
        "source": "UF/IFAS Extension",
    },
    "Potato___Early_blight": {
        "url": "https://extension.umaine.edu/ipm/ipddl/publications/5087e/",
        "source": "University of Maine Cooperative Extension",
    },
    "Potato___Late_blight": {
        "url": "https://extension.psu.edu/tomato-potato-late-blight-in-the-home-garden",
        "source": "Penn State Extension",
    },
    "Tomato_Bacterial_spot": {
        "url": "https://extension.umd.edu/resource/bacterial-diseases-tomato",
        "source": "University of Maryland Extension",
    },
    "Tomato_Early_blight": {
        "url": "https://extension.umaine.edu/ipm/ipddl/publications/5087e/",
        "source": "University of Maine Cooperative Extension",
    },
    "Tomato_Late_blight": {
        "url": "https://www.vegetables.cornell.edu/crops/tomatoes/late-blight/",
        "source": "Cornell Vegetables",
    },
    "Tomato_Leaf_Mold": {
        "url": "https://www.vegetables.cornell.edu/pest-management/disease-factsheets/tomato-leaf-mold/",
        "source": "Cornell Vegetables",
    },
    "Tomato_Septoria_leaf_spot": {
        "url": "https://extension.umd.edu/resource/septoria-leaf-spot-tomatoes",
        "source": "University of Maryland Extension",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "url": "https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/two-spotted-spider-mites/",
        "source": "University of Delaware Cooperative Extension",
    },
    "Tomato__Target_Spot": {
        "url": "https://edis.ifas.ufl.edu/publication/PP351",
        "source": "UF/IFAS Extension",
    },
    "Tomato__Tomato_mosaic_virus": {
        "url": "https://u.osu.edu/hightunneldiseasefacts/tomato-diseases/tomato-mosaic/",
        "source": "Ohio State University Extension",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "url": "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/",
        "source": "UC IPM",
    },
}


class _TextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)


def _html_to_text(html, max_chars=4000):
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"\s+", " ", " ".join(parser.chunks)).strip()
    return text[:max_chars]


def fetch(category, timeout=25):
    """Fetch and clean one extension fact sheet. Returns a RAG-style document or None."""
    entry = EXTENSION_SOURCES.get(category)
    if entry is None:
        return None

    response = requests.get(entry["url"], headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    text = _html_to_text(response.text)

    return {
        "id": f"{category}__extension",
        "text": text,
        "metadata": {"source": entry["source"], "category": category, "url": entry["url"]},
    }


def fetch_all():
    docs = []
    for category in EXTENSION_SOURCES:
        try:
            doc = fetch(category)
            if doc:
                docs.append(doc)
        except requests.RequestException as e:
            print(f"skipped {category}: {e}")
    return docs
