"""Smoke test: run a handful of conversation patterns against the local API
and print results. Use after `uvicorn main:app --port 8000`."""

import json
import sys

import requests

BASE = "http://localhost:8000"


def call(messages: list[dict]) -> dict:
    r = requests.post(f"{BASE}/chat", json={"messages": messages}, timeout=60)
    r.raise_for_status()
    return r.json()


def pretty(label: str, resp: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"reply: {resp['reply'][:300]}")
    print(f"end_of_conversation: {resp['end_of_conversation']}")
    print(f"recommendations ({len(resp['recommendations'])}):")
    for rec in resp["recommendations"]:
        print(f"  - {rec['name']} [{rec['test_type']}]  {rec['url']}")


def main() -> int:
    print("Health:", requests.get(f"{BASE}/health", timeout=10).json())

    # 1. Vague turn 1 — should CLARIFY (empty recs)
    pretty("Vague query (should clarify)", call([
        {"role": "user", "content": "I need an assessment"},
    ]))

    # 2. Java developer — should RECOMMEND
    pretty("Java developer (should recommend)", call([
        {"role": "user", "content": "Hiring a senior Java developer who works with Spring and SQL"},
    ]))

    # 3. Off-topic — should REFUSE
    pretty("Off-topic (should refuse)", call([
        {"role": "user", "content": "What's the capital of France?"},
    ]))

    # 4. Comparison
    pretty("Comparison", call([
        {"role": "user", "content": "What is the difference between OPQ32r and Verify G+?"},
    ]))

    # 5. Confirmation triggers end
    pretty("Confirmation closes conversation", call([
        {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
        {"role": "assistant", "content": "What is the seniority level?"},
        {"role": "user", "content": "Mid-level, around 4 years"},
        {"role": "assistant", "content": "Here are 3 assessments that fit."},
        {"role": "user", "content": "Perfect, that's what we need."},
    ]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
