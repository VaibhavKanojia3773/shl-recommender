# SHL Conversational Assessment Recommender

A FastAPI service that recommends SHL Individual Test Solutions through multi-turn dialogue.

## Endpoints

- `GET /health` → `{"status": "ok", "catalog_size": N}`
- `POST /chat` — body: `{"messages": [{"role": "user|assistant", "content": "..."}]}`. Returns `{"reply": "...", "recommendations": [{"name": "...", "url": "...", "test_type": "K"}], "end_of_conversation": false}`.

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM (main) | Groq `llama-3.3-70b-versatile` | Free tier, fast (~1-2s), strong JSON output |
| LLM (intent) | Groq `llama-3.1-8b-instant` | Sub-second keyword extraction |
| Vector store | ChromaDB (persistent) | Embedded, zero ops, ships with image |
| Embeddings | `all-MiniLM-L6-v2` | 22 MB, no API key, baked into Docker |
| API | FastAPI + Uvicorn | Async, Pydantic schemas |
| Deploy | Render (free Docker tier) | Cold start ≤90 s, fits 2 min `/health` allowance |

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python download_catalog.py        # → catalog.json
python build_vectorstore.py       # → chroma_db/

copy .env.example .env            # then fill in GROQ_API_KEY
uvicorn main:app --reload --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a senior Java developer\"}]}"
```

## Deploy to Render

1. Commit everything (including `catalog.json` and `chroma_db/`) and push to GitHub.
2. render.com → New → Web Service → Connect repo.
3. Docker runtime is auto-detected.
4. Set `GROQ_API_KEY` in the Environment tab.
5. Deploy. Submit `https://<service>.onrender.com` as the endpoint.

## Agent behavior

| Mode | Trigger |
|---|---|
| CLARIFY | Vague turn-1 query (no role, no skills, no level) |
| RECOMMEND | Enough context — pick 1-10 catalog items |
| REFINE | "Drop X / add Y / must be in Spanish" |
| COMPARE | "What's the difference between A and B?" |
| REFUSE | Off-topic, legal interpretation, prompt injection |

Hallucination guard: every returned URL is checked against `catalog.json`. Unknown URLs are silently dropped.

## Layout

```
download_catalog.py     # one-time: fetch the provided JSON catalog
build_vectorstore.py    # one-time: embed into ChromaDB
agent.py                # RAG pipeline, prompts, parser, validator
main.py                 # FastAPI app
catalog.json            # ground-truth catalog (committed)
chroma_db/              # persistent vector index (committed)
Dockerfile              # production image
render.yaml             # Render config
```
