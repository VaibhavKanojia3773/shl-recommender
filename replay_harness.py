"""SHL-style replay harness.

Simulates the evaluator's behaviour: for each of the ten public traces
(C1–C10), build a persona-driven user that answers the agent's questions
from a fact set and ends the conversation when a shortlist is offered.
Score schema compliance, the 8-turn cap, hallucinations, and Recall@10
against the labelled expected shortlist from each sample.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

# Force stdout to UTF-8 so agent replies that contain en-dashes, ellipses, or
# other non-CP1252 characters don't crash the run on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = "https://shl-recommender-2dxd.onrender.com"
CATALOG = json.loads(Path("catalog.json").read_text(encoding="utf-8"))
VALID_URLS = {item["link"] for item in CATALOG}
MAX_TURNS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Traces. Each trace has:
#   persona     — short description (for printing)
#   facts       — list of (trigger_keywords, answer) pairs. When the agent's
#                 most recent reply contains any trigger keyword, the simulated
#                 user answers with the paired text.
#   opening     — the user's first message
#   followups   — fallback messages used in order when no fact-trigger fires
#   expected    — ground-truth URLs the final shortlist should contain
# ─────────────────────────────────────────────────────────────────────────────

TRACES = [
    {
        "id": "C1",
        "persona": "Senior leadership selection (CXO / director level)",
        "opening": "We need a solution for senior leadership.",
        "facts": [
            (["who", "for whom", "meant for", "what level", "seniority", "role"],
             "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."),
            (["selection", "development", "newly created", "in role", "purpose"],
             "Selection — comparing candidates against a leadership benchmark."),
        ],
        "followups": ["Yes, please proceed.", "No particular preference.", "That works."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
            "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
        ],
    },
    {
        "id": "C2",
        "persona": "Senior Rust engineer (no exact catalog match)",
        "opening": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
        "facts": [
            (["cognitive", "verify g+", "general reasoning", "reasoning"],
             "Yes, please add a cognitive test."),
            (["personality", "opq32r", "behavioural", "opq"],
             "Yes, include OPQ32r."),
            (["shortlist", "go ahead", "build", "want me to"],
             "Yes, go ahead. Should I also add a cognitive test for this level?"),
        ],
        "followups": ["That works. Thanks."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
            "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
            "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C3",
        "persona": "Entry-level contact-centre, US English",
        "opening": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
        "facts": [
            (["language", "spoken", "callers"], "English."),
            (["accent", "us", "uk", "australian", "indian", "variant", "fits your operation"], "US."),
            (["different", "differs", "difference", "contact center", "customer service phone"],
             "Perfect — new simulation for volume, old solution for finalists. Confirmed."),
        ],
        "followups": ["Confirmed. Thanks."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
            "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
            "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
            "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
        ],
    },
    {
        "id": "C4",
        "persona": "Graduate financial analysts",
        "opening": "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
        "facts": [
            (["situational", "scenarios", "judgement", "judgment"],
             "Good. Can you also add a situational judgement element — work-context decision making for graduates?"),
        ],
        "followups": ["That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
            "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
            "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C5",
        "persona": "Sales re-skilling / annual talent audit",
        "opening": "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
        "facts": [
            (["opq mq", "sales report", "mq", "motivation"],
             "What's the difference between OPQ and OPQ MQ Sales Report?"),
        ],
        "followups": [
            "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.",
        ],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
            "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
            "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
        ],
    },
    {
        "id": "C6",
        "persona": "Safety-critical plant operators (chemical facility)",
        "opening": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?",
        "facts": [
            (["dsi", "safety & dependability", "8.0", "differ", "difference"],
             "What's the difference between the DSI and the Safety & Dependability 8.0?"),
            (["industrial", "manufacturing", "industry", "norms", "sector"],
             "We're industrial. The 8.0 bundle is the right fit. Confirmed."),
        ],
        "followups": ["Confirmed."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
            "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
        ],
    },
    {
        "id": "C7",
        "persona": "Senior full-stack engineer (Java/Spring/SQL/AWS/Docker)",
        "opening": (
            "Here's the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"
            "Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, "
            "AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, "
            "and mentor mid-level engineers. Strong CI/CD and cloud-native experience required."
        ),
        "facts": [
            (["backend", "frontend", "balanced", "lean"],
             "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features."),
            (["seniority", "senior ic", "tech lead", "ic", "lead", "manage"],
             "Senior IC. They lead design on their own services but don't manage other engineers directly."),
            (["advanced", "entry-level", "right pick", "level"],
             "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?"),
            (["verify g+", "redundant", "cognitive", "leaner"],
             "Keep Verify G+. Locking it in."),
        ],
        "followups": [
            "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.",
            "Locking it in.",
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
    {
        "id": "C8",
        "persona": "Bilingual healthcare admin (Spanish/English, HIPAA)",
        "opening": "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?",
        "facts": [
            (["hybrid", "bilingual", "english fluent", "personality-only", "trade", "fits"],
             "They're functionally bilingual — English fluent for written work. Go with the hybrid."),
            (["hipaa", "legal", "required", "satisfy"],
             "Understood. Keep the shortlist as-is."),
        ],
        "followups": ["Keep the shortlist as-is."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/hipaa-security/",
            "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
            "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C9",
        "persona": "Admin assistants — Excel + Word",
        "opening": "I need to quickly screen admin assistants for Excel and Word daily.",
        "facts": [
            (["simulation", "simulations", "skip personality", "capabilit"],
             "In that case, I am OK with adding a simulation - we want to capture the capabilities."),
        ],
        "followups": ["That's good."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
            "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
            "https://www.shl.com/products/product-catalog/view/ms-word-new/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C10",
        "persona": "Graduate management trainee battery — wants OPQ alternative",
        "opening": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
        "facts": [
            (["opq32r", "shorter", "alternative", "remove", "replace"],
             "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."),
        ],
        "followups": ["Final list: Verify G+ and Graduate Scenarios."],
        "expected": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Schema and behaviour checks
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(resp, where=""):
    problems = []
    if not isinstance(resp, dict):
        return [f"{where}: not a JSON object"]
    keys = set(resp.keys())
    if keys != {"reply", "recommendations", "end_of_conversation"}:
        problems.append(f"{where}: unexpected keys {keys}")
    if not isinstance(resp.get("reply"), str):
        problems.append(f"{where}: reply not a string")
    if not isinstance(resp.get("end_of_conversation"), bool):
        problems.append(f"{where}: end_of_conversation not bool")
    recs = resp.get("recommendations")
    if not isinstance(recs, list):
        problems.append(f"{where}: recommendations not a list")
        return problems
    if len(recs) > 10:
        problems.append(f"{where}: more than 10 recs ({len(recs)})")
    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            problems.append(f"{where}.rec[{i}]: not a dict")
            continue
        if set(rec.keys()) != {"name", "url", "test_type"}:
            problems.append(f"{where}.rec[{i}]: bad keys {set(rec.keys())}")
        if not isinstance(rec.get("name"), str) or not rec["name"]:
            problems.append(f"{where}.rec[{i}]: bad name")
        url = rec.get("url", "")
        if not isinstance(url, str) or not url:
            problems.append(f"{where}.rec[{i}]: bad url")
        elif url not in VALID_URLS:
            problems.append(f"{where}.rec[{i}]: URL not in catalog: {url}")
        if not isinstance(rec.get("test_type"), str):
            problems.append(f"{where}.rec[{i}]: test_type not a string")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Simulated user
# ─────────────────────────────────────────────────────────────────────────────

CONFIRMATIONS = (
    "perfect", "confirmed", "thanks", "that's it", "that's what we need",
    "locking it in", "looks good", "sounds good", "good two-stage",
)


def pick_user_reply(trace: dict, agent_reply: str, used_facts: set, followup_idx: list) -> str:
    """Return the next user message based on the agent's reply."""
    text = (agent_reply or "").lower()
    for i, (triggers, answer) in enumerate(trace["facts"]):
        if i in used_facts:
            continue
        if any(t in text for t in triggers):
            used_facts.add(i)
            return answer
    if followup_idx[0] < len(trace["followups"]):
        msg = trace["followups"][followup_idx[0]]
        followup_idx[0] += 1
        return msg
    return "No preference."


def looks_like_shortlist_offered(resp: dict) -> bool:
    """The simulated user ends the chat once the agent gives a shortlist."""
    return len(resp.get("recommendations", [])) >= 1


def call_chat(messages, timeout=60, retries=3):
    last_exc = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(f"{BASE}/chat", json={"messages": messages}, timeout=timeout)
            elapsed = time.time() - t0
            r.raise_for_status()
            return r.json(), elapsed
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            sleep_for = 2 ** attempt
            print(f"   [retry {attempt + 1}/{retries}] {type(exc).__name__}; sleeping {sleep_for}s")
            time.sleep(sleep_for)
    raise last_exc  # type: ignore[misc]


def run_trace(trace: dict, verbose: bool = True) -> dict:
    """Run one persona-driven conversation. Returns a result dict."""
    messages = [{"role": "user", "content": trace["opening"]}]
    used_facts = set()
    followup_idx = [0]

    schema_problems: list[str] = []
    per_turn_latencies: list[float] = []
    last_resp = None
    final_recs: list[dict] = []
    end_signaled_by_agent = False
    turn_index = 0

    if verbose:
        print(f"\n── {trace['id']}: {trace['persona']} ──")

    while True:
        # Enforce simulated-user discipline: don't exceed MAX_TURNS user messages.
        if sum(1 for m in messages if m["role"] == "user") > MAX_TURNS:
            break
        turn_index += 1
        if turn_index > MAX_TURNS:
            break

        resp, elapsed = call_chat(messages)
        per_turn_latencies.append(elapsed)
        last_resp = resp

        problems = validate_schema(resp, where=f"{trace['id']}.t{turn_index}")
        schema_problems.extend(problems)

        if verbose:
            n = len(resp.get("recommendations", []))
            print(f"  user→ {messages[-1]['content'][:80]}")
            print(f"  bot (t={turn_index}, {elapsed:.1f}s, recs={n}, end={resp['end_of_conversation']}): "
                  f"{resp['reply'][:120]}")

        if resp.get("recommendations"):
            final_recs = resp["recommendations"]

        # Stop if the agent signals end_of_conversation
        if resp.get("end_of_conversation"):
            end_signaled_by_agent = True
            break

        # If the agent offered a shortlist, the simulated user typically confirms.
        # This mirrors the assignment's note: "ends the conversation when the
        # agent provides a shortlist."
        if looks_like_shortlist_offered(resp) and turn_index >= 2:
            # Sometimes there's still a useful clarification (compare / refine).
            # Use the persona's next fact if it fires on this reply; otherwise
            # confirm and exit.
            text_lower = (resp.get("reply") or "").lower()
            next_fact = None
            for i, (triggers, answer) in enumerate(trace["facts"]):
                if i in used_facts:
                    continue
                if any(t in text_lower for t in triggers):
                    next_fact = answer
                    used_facts.add(i)
                    break
            if next_fact:
                messages.append({"role": "assistant", "content": resp["reply"]})
                messages.append({"role": "user", "content": next_fact})
                continue
            # Confirm and end
            messages.append({"role": "assistant", "content": resp["reply"]})
            messages.append({"role": "user", "content": "Perfect, that's what we need."})
            confirm_resp, elapsed = call_chat(messages)
            per_turn_latencies.append(elapsed)
            problems = validate_schema(confirm_resp, where=f"{trace['id']}.confirm")
            schema_problems.extend(problems)
            last_resp = confirm_resp
            if confirm_resp.get("recommendations"):
                final_recs = confirm_resp["recommendations"]
            end_signaled_by_agent = bool(confirm_resp.get("end_of_conversation"))
            turn_index += 1
            break

        # Otherwise continue dialogue with a persona-driven user message
        next_msg = pick_user_reply(trace, resp["reply"], used_facts, followup_idx)
        messages.append({"role": "assistant", "content": resp["reply"]})
        messages.append({"role": "user", "content": next_msg})

    # Score Recall@10 against the labelled expected set
    expected = set(trace["expected"])
    returned_urls = {rec["url"] for rec in final_recs}
    hit_urls = expected & returned_urls
    recall_at_10 = (len(hit_urls) / len(expected)) if expected else 0.0

    # Turn cap check: user-turn count
    user_turns = sum(1 for m in messages if m["role"] == "user")

    result = {
        "id": trace["id"],
        "persona": trace["persona"],
        "user_turns": user_turns,
        "agent_set_end_of_conversation": end_signaled_by_agent,
        "schema_problems": schema_problems,
        "final_recs": final_recs,
        "hit_urls": sorted(hit_urls),
        "missed_urls": sorted(expected - returned_urls),
        "extra_urls": sorted(returned_urls - expected),
        "recall_at_10": recall_at_10,
        "avg_latency": (sum(per_turn_latencies) / len(per_turn_latencies)) if per_turn_latencies else 0.0,
        "max_latency": max(per_turn_latencies) if per_turn_latencies else 0.0,
    }

    if verbose:
        print(f"  → user_turns={user_turns}, recall@10={recall_at_10:.2%}, "
              f"avg_latency={result['avg_latency']:.1f}s, max={result['max_latency']:.1f}s, "
              f"schema_problems={len(schema_problems)}, end={end_signaled_by_agent}")
        if hit_urls:
            print(f"  HITS:   {len(hit_urls)}/{len(expected)}")
            for u in sorted(hit_urls):
                print(f"     [+] {u}")
        if expected - returned_urls:
            print(f"  MISSES: {len(expected - returned_urls)}/{len(expected)}")
            for u in sorted(expected - returned_urls):
                print(f"     [-] {u}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Health check first (the evaluator does this, with 2-min grace)
    print(f"GET {BASE}/health")
    h = requests.get(f"{BASE}/health", timeout=120).json()
    print(f"   {h}")
    assert h.get("status") == "ok", "health check failed"

    results = []
    for trace in TRACES:
        try:
            results.append(run_trace(trace, verbose=True))
        except Exception as exc:
            print(f"  !! {trace['id']} crashed: {type(exc).__name__}: {exc}")
            results.append({
                "id": trace["id"], "persona": trace["persona"],
                "schema_problems": [f"crash: {exc}"], "recall_at_10": 0.0,
                "user_turns": 0, "agent_set_end_of_conversation": False,
                "hit_urls": [], "missed_urls": trace["expected"], "extra_urls": [],
                "avg_latency": 0.0, "max_latency": 0.0, "final_recs": [],
            })

    # Aggregate
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'ID':4} {'Recall@10':>10}  {'Turns':>6}  {'End?':5}  {'AvgLat':>7}  {'MaxLat':>7}  {'Schema':>7}"
    print(header)
    print("-" * 78)
    total_recall = 0.0
    total_problems = 0
    for r in results:
        print(f"{r['id']:4} {r['recall_at_10']*100:9.1f}%  "
              f"{r['user_turns']:6d}  {str(r['agent_set_end_of_conversation']):5}  "
              f"{r['avg_latency']:6.1f}s  {r['max_latency']:6.1f}s  "
              f"{len(r['schema_problems']):>7}")
        total_recall += r["recall_at_10"]
        total_problems += len(r["schema_problems"])
    print("-" * 78)
    mean_recall = total_recall / len(results) if results else 0.0
    print(f"\nMean Recall@10 across {len(results)} traces: {mean_recall:.2%}")
    print(f"Total schema problems: {total_problems}")
    print(f"All traces under 8 user-turns: "
          f"{all(r['user_turns'] <= MAX_TURNS for r in results)}")
    print(f"All max-latency under 30s: "
          f"{all(r['max_latency'] <= 30 for r in results)}")

    # Persist JSON report
    Path("replay_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nFull report written to replay_report.json")

    # Exit non-zero only on hard fails (schema or turn cap)
    hard_fail = (total_problems > 0 or
                 any(r["user_turns"] > MAX_TURNS for r in results))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
