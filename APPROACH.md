# SHL Conversational Assessment Recommender — Approach

**Vaibhav Kanojia** · vaibhavkanojia3773@gmail.com
Endpoint: https://shl-recommender-2dxd.onrender.com
Source: https://github.com/VaibhavKanojia3773/shl-recommender

---

### Design choices

The catalog is closed (377 Individual Test Solutions). The agent should never invent an assessment. That single constraint shapes the whole system: retrieval first, generation second, then a hard validator that drops anything the model fabricates. The stack is FastAPI for the API surface, ChromaDB (persistent, embedded) for vectors, `all-MiniLM-L6-v2` for embeddings, and Groq's free tier for the LLMs — Llama 3.3 70B Versatile as the main generator with Llama 3.1 8B Instant as a quota-fallback and as the intent extractor. Render's free Docker tier hosts it. The 2-minute cold-start window the assignment allows on `/health` matches Render's free-tier wake time, so no infrastructure compromises were needed.

Two models, not one, was a deliberate split. Asking the 70B to also extract search keywords added 1–2 s of latency per turn for no quality gain; the 8B model produces a clean keyword string in ~300 ms and frees the larger model to focus on reasoning and JSON output. The `chroma_db/` directory and `catalog.json` are both committed to the repo and baked into the Docker image so cold starts don't re-embed the catalog or download anything at runtime.

### Retrieval setup

A single semantic query over a multi-aspect job description (e.g. *"Java + Spring + AWS + Docker + senior IC"*) misses things. The dominant embedding direction wins; the others get drowned. So the retriever splits the intent string into aspect-specific sub-queries triggered by keywords (`java`, `aws`, `docker`, `leadership`, `graduate`, `contact center`, `hipaa`, `safety`, …) and unions the top hits from each. A small set of default-battery seeds — OPQ32r, Verify G+, Graduate Scenarios, DSI, Core Java Advanced — is injected at the top of the candidate pool when their triggers fire, mirroring the layering pattern the sample conversations consistently use. The pool is capped at 22 items so the LLM sees a wide, targeted slate without bloating the prompt. Offline retrieval analysis confirmed that this lifts top-K recall on the multi-aspect personas from ~62 % to ~98 % without any additional LLM calls.

### Prompt design

The system prompt locks the model into one of five modes — **CLARIFY**, **RECOMMEND**, **REFINE**, **COMPARE**, **REFUSE** — and gives it explicit decision rules for choosing between them. Four absolute rules sit at the top: catalog-only, verbatim URLs, refuse off-topic and legal interpretation, don't recommend on a vague turn 1. The 22 retrieved candidates are appended as a structured numbered list with full metadata (name, URL, test-type codes, keys, job levels, languages, duration, 200-char description). Output shape is enforced two ways — the prompt shows a concrete JSON example, and Groq's `response_format={"type": "json_object"}` guarantees JSON at the API level. The reply is parsed through a four-stage fallback (direct JSON → markdown-fenced → regex-extracted → safe default), and every URL is validated against `catalog.json` before it reaches the client. If the model returns a real catalog name but a malformed URL, the validator recovers the canonical URL by name lookup. Hallucinated URLs are silently dropped — they never reach the user.

### Evaluation approach

Two harnesses run against the live endpoint. `production_audit.py` exercises targeted probes: `/health`, vague-turn-1 (must clarify), clear JD (must recommend), off-topic and code-writing requests (must refuse), prompt-injection with a fabricated URL (must drop), a non-existent assessment name (must drop), comparison, refinement with explicit drops, confirmation closure, and an empty-body request (must 4xx). `replay_harness.py` simulates the SHL evaluator more faithfully: a persona-driven user that answers the agent's questions from a fact set, says *"no preference"* outside its facts, and ends when a shortlist is offered — the exact behaviour described in the assignment. The harness records per-trace schema problems, hallucination counts, turn count, latency, and Recall@10 against the labelled expected shortlist. Both ran continuously while I iterated.

### What didn't work, and how I measured improvement

A single concatenated query through ChromaDB was the first thing I tried. The replay harness flagged C7 (multi-stack Java engineer) and C9 (Excel + Word admin) as systematic 0 % traces. Offline analysis showed the right URLs sat at rank 30–60 — present, but unreachable in a top-20 pool. Aspect-based sub-queries plus seeded defaults moved those traces from 0 % toward parity with the simpler personas, with measurable lifts visible in the harness's per-trace Recall@10. Widening the candidate pool further to 30 then *hurt*, because prompt token growth pushed P50 latency past the 25-second internal timeout — I caught this in the same harness's latency column and trimmed back to 22 with shorter descriptions.

A separate detour: I migrated the main model to Gemini 2.0 Flash to escape Groq's 100 k tokens-per-day cap. Every Gemini key I generated landed on a Google Cloud project with free-tier quota set to zero (HTTP 429 `RESOURCE_EXHAUSTED` on the first call) or denied access entirely (HTTP 403). A Google-side issue, not a code one. Rolled back to Groq with a graceful 70B → 8B fallback to keep the service resilient when the 70B's daily cap is hit.

Filtering ChromaDB by test-type codes when the user said *"cognitive"* or *"personality"* hurt recall measurably — a *"cognitive"* query should still surface relevant simulation- and biodata-type items. The LLM filters from a broader pool better than the retriever does.

### Use of AI tools

I used **Claude Code** (Anthropic) as a coding assistant throughout: scaffolding the FastAPI service and Dockerfile, debugging the CPU-only PyTorch deploy on Render, iterating the system prompt, building the replay harness, and writing this document. Design decisions — the retrieval-first architecture, the two-model split, the seeded multi-aspect pool, the validator-as-safety-net, the mode-driven prompt — were mine.

---

### System diagram

![Architecture](architecture.png)
