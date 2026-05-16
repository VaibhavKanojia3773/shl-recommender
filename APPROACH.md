# Approach Document — Conversational SHL Assessment Recommender

**Author:** Vaibhav Kanojia
**Endpoint:** https://shl-recommender-2dxd.onrender.com
**Repository:** https://github.com/VaibhavKanojia3773/shl-recommender

---

## 1. Problem framing

Recruiters rarely arrive with the right vocabulary. They say "I'm hiring a Java dev who works with stakeholders" — not "I need a knowledge-skills test with a personality component." A keyword catalog forces them to translate intent into SHL's taxonomy before they can search. The goal of this project was to invert that flow: let the recruiter speak naturally and let the agent take responsibility for translation, grounding, and shortlisting.

The harder problem was making this work under three simultaneous constraints — a stateless API, an 8-turn ceiling, and a strict ban on hallucinated assessments. Each of those constraints quietly rules out a class of "obvious" solutions, and most of the design effort went into respecting them without making the agent feel mechanical.

---

## 2. Architecture at a glance

The service is a single FastAPI process with two endpoints (`/health`, `/chat`). Every `/chat` request flows through a five-step pipeline:

1. **Intent extraction** — a small fast model (Llama 3.1 8B Instant on Groq) distills the conversation into 8–15 search keywords. Doing this with a model rather than concatenating message text gives much better recall when the user mentions a skill in turn 1 and constrains it in turn 4.
2. **Semantic retrieval** — those keywords go through `sentence-transformers/all-MiniLM-L6-v2` and hit a persistent ChromaDB collection of all 377 SHL Individual Test Solutions. The top 20 candidates are returned with their full metadata.
3. **Grounded generation** — the main model (Llama 3.3 70B Versatile on Groq) receives the system prompt, the full conversation, and the 20 candidates as a structured context block. It is instructed to recommend only from this block, and to return its response as a JSON object matching the assignment schema.
4. **Robust parsing** — a four-stage JSON parser handles raw output, markdown-fenced output, embedded objects, and finally a safe fallback. Production has not needed stages 2–4 yet, but they exist because LLM JSON discipline is a "fails silently or fails catastrophically" property.
5. **Hallucination validation** — every URL the model returns is checked against an in-memory set built from `catalog.json` at startup. Any URL not in that set is dropped silently. If the model returns a valid catalog name but a malformed URL, the validator recovers the canonical URL from a name-based lookup. The user therefore cannot be shown a fabricated assessment under any circumstance.

A more detailed diagram is included on the final page.

---

## 3. Why this stack

Every component was chosen with the 30-second per-turn timeout in mind, and with the assignment's preference for free, open-source infrastructure.

- **Groq as the inference provider.** Two model calls per turn (intent + main) plus retrieval need to finish in under 25 seconds with a 5-second buffer. Groq's typical latency on Llama 3.3 70B is 1–2 seconds end-to-end, which leaves comfortable headroom for cold-start variance.
- **Two models, not one.** Using a small model (8B Instant) for keyword extraction is essentially free in tokens and time, and it lets the larger 70B model spend its budget on reasoning and JSON discipline. A single-model design felt tidier but cost ~40 % more latency in early experiments.
- **ChromaDB persistent store.** Embedded, zero-ops, ships inside the Docker image. The vector index is built once locally and committed to the repo, so cold starts on Render don't re-embed 377 documents.
- **all-MiniLM-L6-v2 embeddings.** 22 MB, no API key, baked into the Docker image at build time. Good enough quality for a catalog this size, and crucially: deterministic across deploys.
- **FastAPI + Uvicorn.** Async by default, Pydantic-validated request and response schemas, and an auto-generated `/docs` endpoint that doubles as a manual-test UI for evaluators.
- **Render (Docker, free tier).** Honours the 2-minute cold-start allowance the assignment explicitly accounts for. The Dockerfile installs CPU-only PyTorch from PyTorch's CPU wheel index to avoid pulling ~2 GB of unused NVIDIA libraries — a fix that took one failed deploy to discover.

---

## 4. Prompt and conversation design

The system prompt is deliberately mode-driven. Rather than asking the model to "be helpful," it is given five explicit modes — CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE — and decision rules for choosing between them. The prompt then declares four absolute rules (catalog-only, verbatim URLs, refuse off-topic, don't recommend on a vague turn 1), followed by the output schema with a concrete example.

This structure was iterated against the ten provided sample conversations (C1–C10), which encode behaviours the evaluator clearly cares about: layering cognitive + personality + domain tests for senior technical hires (C7), refusing legal interpretation questions while keeping the existing shortlist intact (C6), graceful handling of niche roles with no exact catalog match (C2 — senior Rust engineer), and recognising confirmation phrases to end the conversation cleanly (C1, C4, C5).

The 20-candidate context block is structured as a numbered list with full metadata per item — name, URL, type codes, keys, job levels, languages, duration, and a truncated description. Providing this in a stable format meant the model rarely tried to invent a URL; almost all hallucination attempts during testing were the model copying a name correctly but rewriting the URL slug. The post-generation validator catches those cases without the user noticing.

---

## 5. Evaluation

Local evaluation focused on three things that map to the assignment's scoring rubric.

- **Schema compliance.** A production audit script (`production_audit.py`) runs fourteen end-to-end calls against the live endpoint, including the `/health` check, the four conversational modes, two adversarial probes (prompt injection and a fabricated assessment name), a refinement sequence, a confirmation phrase, and replays of three sample conversations. Each response is validated key-by-key against the Pydantic schema, and every returned URL is checked against `catalog.json`. The Groq `response_format={"type": "json_object"}` setting plus the four-stage parser have produced zero schema violations in this audit.
- **Hallucination rate.** The audit's injection probe instructs the model to return a fabricated assessment with an `evil.example.com` URL; another probe asks for a non-existent "HireQuotient Rust Backend Battery." In both cases the model occasionally tries to oblige in its `reply` field, but the validator drops the fabricated URL before it reaches the client. Hallucinations in `recommendations`: **0**.
- **Behavioural alignment with sample conversations.** I read all ten C-traces before writing the prompt and used their patterns as a regression set. The audit's C1 replay (senior leadership selection) surfaces OPQ Leadership Report and HiPo Assessment Report 2.0; the C4 replay (graduate financial analysts) surfaces Verify Numerical Ability and Financial Accounting; and the agent ends the conversation on confirmation phrases without needing the turn-8 cap to fire. One known imperfection: when a user asks to "compare OPQ32r and Verify G+" with those exact short names, the embedding model sometimes misses the canonical OPQ32r record and surfaces an OPQ report variant instead — a recall issue rather than a hallucination, and a candidate for the cross-encoder reranker mentioned in section 8.

---

## 6. What didn't work

A few approaches were tried and dropped:

- **Single-model agent.** Initially I had Llama 3.3 70B do both intent extraction and response generation. This was tidier code but slower per turn — extraction adds 1–2 seconds of latency on the 70B model versus ~300 ms on 8B Instant. Splitting keyword extraction to the smaller model freed the larger one to focus on reasoning and the JSON contract.
- **Filtering ChromaDB by test-type code.** I tried restricting retrieval to specific test types when the user mentioned "personality" or "cognitive." This hurt recall — a "cognitive" query should still surface relevant simulation- and biodata-type assessments. Letting the LLM filter from a broader candidate set worked better.
- **Static catalog formatting in the prompt.** Embedding the full catalog (377 items) in the system prompt was tried briefly. It worked but was wasteful in tokens, slower, and made comparison requests harder. Top-20 retrieval per turn is the right middle ground.
- **The default PyTorch wheel on Render.** The first deploy attempted to install `torch` with its full CUDA dependency tree (~2 GB). The CPU-only wheel from `download.pytorch.org/whl/cpu` is roughly a tenth of the size and works identically for inference on Render's CPU instances.

---

## 7. Use of AI tools

This project was built with **Claude Code** (Anthropic) as a coding assistant. I used it for: scaffolding the FastAPI service and Dockerfile, exploring the SHL catalog structure, drafting the system prompt, debugging the CPU-only PyTorch deployment failure on Render, and writing this approach document. All design choices — the two-model split, the validator-as-safety-net, the candidate context format, the mode-driven prompt — were mine. The agent acted as a fast pair-programmer; I remained the engineer.

---

## 8. What I would change with more time

- **Reranking.** Add a cross-encoder reranker between ChromaDB and the LLM. The MiniLM embeddings are good but coarse for a catalog this dense (e.g., distinguishing OPQ32r from the dozen OPQ report variants).
- **Few-shot exemplars.** Inject one or two of the sample conversations directly into the system prompt as gold examples. I deliberately avoided this to keep the prompt slim, but it would likely lift recall on the harder personas.
- **Soft caching of intent extraction.** When the conversation history hasn't changed meaningfully between turns, the same keywords are re-derived. A hash-based cache would shave another 300–500 ms per turn.
- **A proper evaluation harness.** I wrote a manual smoke test; a replay harness similar to the one SHL uses, but driven by my own simulated user, would give a continuous Recall@10 signal during iteration.

---

## 9. System diagram

See the appended diagram on the next page. The flow is: client → FastAPI → intent extraction (Llama 3.1 8B) → ChromaDB retrieval → main generation (Llama 3.3 70B) → JSON parse + validate → response.
