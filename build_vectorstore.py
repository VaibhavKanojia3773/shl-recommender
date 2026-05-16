"""Build a persistent ChromaDB vector store from catalog.json.

Run once locally; commit the resulting chroma_db/ directory to the repo so the
Docker image ships with a ready-to-query index (no re-embedding at cold start).
"""

import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).parent
CATALOG_PATH = ROOT / "catalog.json"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "shl_assessments"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_document_text(item: dict) -> str:
    """Rich text that captures everything searchable about the assessment."""
    keys = ", ".join(item.get("keys") or [])
    job_levels = ", ".join(item.get("job_levels") or [])
    languages = ", ".join(item.get("languages") or [])
    duration = item.get("duration") or "unspecified"
    remote = item.get("remote") or "unknown"
    adaptive = item.get("adaptive") or "unknown"
    description = item.get("description") or ""
    return (
        f"Assessment: {item['name']}\n"
        f"Types: {keys}\n"
        f"Job Levels: {job_levels}\n"
        f"Languages: {languages}\n"
        f"Duration: {duration}\n"
        f"Remote: {remote}\n"
        f"Adaptive: {adaptive}\n"
        f"Description: {description}"
    )


def build_metadata(item: dict) -> dict:
    """ChromaDB metadata values must be str/int/float/bool. Lists are joined."""
    return {
        "entity_id": str(item.get("entity_id") or ""),
        "name": item["name"],
        "url": item["link"],
        "keys": "|".join(item.get("keys") or []),
        "job_levels": "|".join(item.get("job_levels") or []),
        "languages": "|".join(item.get("languages") or []),
        "duration": item.get("duration") or "",
        "remote": item.get("remote") or "",
        "adaptive": item.get("adaptive") or "",
    }


def get_or_build_collection(rebuild: bool = False) -> chromadb.Collection:
    if rebuild and CHROMA_PATH.exists():
        import shutil
        shutil.rmtree(CHROMA_PATH)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def main() -> int:
    if not CATALOG_PATH.exists():
        print(f"ERROR: {CATALOG_PATH} not found. Run download_catalog.py first.", file=sys.stderr)
        return 1

    items = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(items)} assessments from {CATALOG_PATH}")

    rebuild = "--rebuild" in sys.argv
    collection = get_or_build_collection(rebuild=rebuild)

    existing = collection.count()
    if existing == len(items) and not rebuild:
        print(f"Collection already has {existing} items. Use --rebuild to reindex.")
        return 0

    if existing > 0 and not rebuild:
        print(f"Collection has {existing} items; expected {len(items)}. Rebuilding...")
        return main_rebuild()

    ids = [str(item.get("entity_id") or item["link"]) for item in items]
    documents = [build_document_text(item) for item in items]
    metadatas = [build_metadata(item) for item in items]

    print(f"Embedding and indexing {len(items)} assessments...")
    # Add in batches to avoid memory spikes
    batch_size = 64
    for i in range(0, len(items), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  indexed {min(i + batch_size, len(items))}/{len(items)}")

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    return 0


def main_rebuild() -> int:
    sys.argv.append("--rebuild")
    return main()


if __name__ == "__main__":
    sys.exit(main())
