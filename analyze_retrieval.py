"""Offline retrieval analysis. Runs ChromaDB queries without touching Groq
so we can debug recall problems without spending tokens. Prints, for each
trace, which expected URLs would have been in the top-30 retrieval set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = Path("chroma_db")
CATALOG = {item["link"]: item
           for item in json.loads(Path("catalog.json").read_text(encoding="utf-8"))}

# Same wording the agent would see (intent_query). Approximated since the
# real one is LLM-generated; we use the persona's salient nouns.
PROBES = {
    "C1": {
        "queries": [
            "senior leadership selection CXO director executive personality OPQ",
            "leadership benchmark selection executive personality questionnaire",
            "OPQ Universal Competency leadership report",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
            "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
        ],
    },
    "C2": {
        "queries": [
            "senior Rust engineer high performance networking infrastructure systems coding",
            "live coding interview senior software engineer",
            "linux programming general systems",
            "cognitive ability senior technical engineer Verify G",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
            "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
            "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    "C3": {
        "queries": [
            "entry level contact center customer service inbound calls English US",
            "spoken English voice assessment call center",
            "contact center call simulation entry level",
            "customer service phone simulation",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
            "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
            "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
            "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
        ],
    },
    "C4": {
        "queries": [
            "graduate financial analyst numerical reasoning finance knowledge",
            "graduate situational judgment scenarios",
            "basic statistics financial accounting graduate",
            "OPQ32r personality graduate",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
            "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
            "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    "C5": {
        "queries": [
            "sales re-skilling talent audit OPQ sales report motivation",
            "global skills assessment development report",
            "sales transformation individual contributor",
            "OPQ MQ sales report motivators",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
            "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
            "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
        ],
    },
    "C6": {
        "queries": [
            "plant operator chemical safety dependability reliability industrial",
            "safety dependability instrument DSI",
            "workplace health and safety knowledge",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
            "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
        ],
    },
    "C7": {
        "queries": [
            "senior full stack engineer Java Spring REST SQL AWS Docker microservice",
            "core Java advanced knowledge senior engineer",
            "Spring SQL knowledge test senior engineer",
            "AWS development Docker knowledge senior",
            "Verify G+ cognitive senior engineer OPQ32r personality",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
            "https://www.shl.com/products/product-catalog/view/spring-new/",
            "https://www.shl.com/products/product-catalog/view/sql-new/",
            "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
            "https://www.shl.com/products/product-catalog/view/docker-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    "C8": {
        "queries": [
            "bilingual healthcare admin Spanish HIPAA patient records",
            "HIPAA security medical terminology Microsoft Word knowledge",
            "DSI dependability healthcare personality OPQ32r Spanish",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/hipaa-security/",
            "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
            "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    "C9": {
        "queries": [
            "admin assistant Excel Word screen knowledge simulation",
            "MS Excel MS Word knowledge test",
            "Microsoft Excel 365 Microsoft Word 365 simulation",
            "OPQ32r personality admin assistant",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
            "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
            "https://www.shl.com/products/product-catalog/view/ms-word-new/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    "C10": {
        "queries": [
            "graduate management trainee battery cognitive personality situational judgement",
            "Verify G+ cognitive graduate",
            "graduate scenarios situational judgment",
            "OPQ32r personality graduate trainee",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        ],
    },
}


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    coll = client.get_collection(name="shl_assessments", embedding_function=embed)

    print(f"Collection has {coll.count()} items.\n")

    summary = []
    for tid, probe in PROBES.items():
        expected = set(probe["expected"])
        union: set[str] = set()
        for q in probe["queries"]:
            res = coll.query(query_texts=[q], n_results=30)
            for meta in res["metadatas"][0]:
                union.add(meta["url"])
        hit = expected & union
        miss = expected - union
        recall_in_pool = len(hit) / len(expected) if expected else 0
        summary.append((tid, recall_in_pool, len(expected), len(hit), miss))
        print(f"--{tid}: retrieval-recall (top-30 across {len(probe['queries'])} queries) = "
              f"{recall_in_pool:.0%} ({len(hit)}/{len(expected)})")
        for m in sorted(miss):
            item = CATALOG.get(m, {})
            print(f"   MISSED retrieval: {item.get('name', m)}")
            # Show the per-query rank for the missed URL
            for q in probe["queries"]:
                res = coll.query(query_texts=[q], n_results=80)
                urls = [meta["url"] for meta in res["metadatas"][0]]
                if m in urls:
                    print(f"      found at rank {urls.index(m)+1} for query: {q[:80]}")
                    break
            else:
                print(f"      not in top-80 for any query")

    print("\n" + "=" * 70)
    print("SUMMARY (retrieval-only — what the LLM would see if we widened n_results):")
    print("=" * 70)
    for tid, recall, exp, hit, _ in summary:
        print(f"  {tid}: {recall:>4.0%}   ({hit}/{exp} expected URLs were retrievable)")


if __name__ == "__main__":
    main()
