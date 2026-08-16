"""Fetch curated extension fact sheets and cache them locally.

Run this occasionally (e.g. monthly) to refresh the cache -- not on every
RAG pipeline startup, since it hits external sites.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deployment.rag.retrievers.usda import fetch_all

OUT_PATH = os.path.join(os.path.dirname(__file__), "extension_docs.json")


def main():
    docs = fetch_all()
    with open(OUT_PATH, "w") as f:
        json.dump(docs, f, indent=2)
    print(f"cached {len(docs)} extension fact sheets to {OUT_PATH}")


if __name__ == "__main__":
    main()
