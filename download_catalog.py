"""Download the SHL product catalog and save as catalog.json.

Run once locally before building the vector store. The catalog is also
downloaded at Docker build time as a fallback so the image is self-contained.
"""

import json
import sys
from pathlib import Path

import requests

CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)
OUTPUT_PATH = Path(__file__).parent / "catalog.json"


def download() -> list[dict]:
    print(f"Fetching {CATALOG_URL} ...")
    response = requests.get(CATALOG_URL, timeout=60)
    response.raise_for_status()
    # Source JSON contains raw control characters (e.g. newlines inside strings),
    # which json.loads rejects by default. Use strict=False to tolerate them.
    data = json.loads(response.text, strict=False)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    print(f"Received {len(data)} assessments")
    return data


def normalize(items: list[dict]) -> list[dict]:
    """Drop unusable entries and ensure every record has a usable link + name."""
    clean: list[dict] = []
    seen_urls: set[str] = set()
    for item in items:
        name = (item.get("name") or "").strip()
        link = (item.get("link") or "").strip()
        if not name or not link:
            continue
        if link in seen_urls:
            continue
        seen_urls.add(link)
        item["name"] = name
        item["link"] = link
        item["description"] = (item.get("description") or "").strip()
        item["job_levels"] = item.get("job_levels") or []
        item["languages"] = item.get("languages") or []
        item["duration"] = (item.get("duration") or "").strip()
        item["remote"] = (item.get("remote") or "").strip()
        item["adaptive"] = (item.get("adaptive") or "").strip()
        item["keys"] = item.get("keys") or []
        clean.append(item)
    return clean


def main() -> int:
    try:
        raw = download()
    except Exception as exc:
        print(f"ERROR downloading catalog: {exc}", file=sys.stderr)
        return 1

    items = normalize(raw)
    OUTPUT_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} assessments to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
