# SHL Assessment Recommender — Full Documentation

A from-scratch walkthrough for someone who knows ML/AI but is new to web APIs and backend infrastructure.

---

## 1. The big picture: why no website?

When you build an ML model, you usually run `model.predict(x)` from a notebook. That works for you, but **other people / other programs can't call your notebook**.

To let other software talk to your model, you wrap it in an **API** (Application Programming Interface). An API is just a contract:

> "Send me a request shaped like X, and I'll send you a response shaped like Y."

The assignment says:
> *Expose a FastAPI service with two endpoints: GET /health and POST /chat.*

That's the **deliverable**. Not a chatbot website — a machine-readable service. Here's why:

- The SHL evaluator is a **program**, not a human. It runs an automated script that simulates 10+ recruiter conversations against your service and scores how well you recommended assessments. A program can't click buttons on a webpage; it needs an API.
- Building a UI is a separate skill from building the agent. The assignment is testing your **agent design + retrieval + LLM grounding**, not your CSS.
- An API is reusable: tomorrow you could put a chat UI on top of it, integrate it into Slack, or call it from another backend. The agent logic stays the same.

You can absolutely **add a UI later** (I'll show how at the end), but the graded thing is the API.

---

## 2. What is an "endpoint"?

An endpoint is a specific URL on your server that does one thing.

Your service has two:

| Endpoint | What it does |
|---|---|
| `GET  /health` | "Are you alive?" — returns OK |
| `POST /chat`   | "Here's the conversation, what's your next reply?" — runs the agent |

The full URL when running locally is `http://localhost:8000/health`. When deployed it's `https://shl-recommender.onrender.com/health`.

### What's GET vs POST?

These are **HTTP methods** — verbs that say what kind of operation you want:

- **GET** = "fetch something, don't change anything." Like reading a file.
- **POST** = "I'm sending you data, do something with it." Like submitting a form.

`/health` is GET because you just want a status — you're not sending data.
`/chat` is POST because you're sending a conversation history and asking for a reply.

That's the whole distinction. Browsers use GET when you type a URL; POST happens when you submit a form or when a program (like the SHL evaluator) sends JSON.

---

## 3. What does `/health` actually do, and why does it exist?

Here's the entire endpoint:

```python
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "catalog_size": 377}
```

That's it. It returns a tiny JSON. **Why bother?**

Three reasons:

1. **Render needs it.** Render is a free hosting service. When it starts your container, it waits until something on your service responds with HTTP 200. Render hits `/health` repeatedly. As soon as it gets `200 OK`, Render marks your service "live" and starts sending real traffic. Without this, Render would never know when you're ready.

2. **The SHL evaluator needs it.** The assignment says:
   > *For cold start hosting services, the first /health call will allow up to 2 minutes for service to wake up.*

   The evaluator pings `/health` before submitting `/chat` requests. If `/health` returns OK, the evaluator knows your service has finished loading the catalog and ChromaDB.

3. **You need it.** When you change something and redeploy, hitting `/health` is the simplest "did I break something" check.

**Concrete example:** Open http://localhost:8000/health in your browser right now (while the server is running). You'll see:
```json
{"status": "ok", "catalog_size": 377}
```

That's the endpoint. There's no UI. The browser is showing you raw JSON because that's all the server sent. The SHL evaluator parses this same JSON.

---

## 4. What does `/chat` actually do?

This is the brain. Here's the contract:

### Request
The client (evaluator / your test script / a future UI) sends a **POST** with a JSON body:
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

Note: **the whole conversation is sent every time**. This is called "stateless." Your server doesn't remember anything between calls. The client carries the history.

### Response
Your server returns JSON shaped exactly like:
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev.",
  "recommendations": [
    {"name": "Core Java (Advanced Level)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Spring (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

Three fields, every time:
- **reply** — the chatbot's text answer to the user.
- **recommendations** — 0 to 10 SHL assessments (empty when clarifying or refusing).
- **end_of_conversation** — `true` when the user said "thanks, done" or we hit turn 8.

**The schema is non-negotiable.** The assignment doc says:
> *Deviating breaks our automated evaluator, and your submission will not score.*

That's why we use Pydantic models — they enforce the shape automatically.

---

## 5. End-to-end flow (what happens on a single `/chat` call)

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLIENT (evaluator script, smoke_test.py, or future UI)             │
│                                                                     │
│  POST http://localhost:8000/chat                                    │
│  Body: { "messages": [...] }                                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                                  │
│                                                                     │
│  1. Receive the JSON request                                        │
│  2. Validate with Pydantic — reject malformed requests with 400     │
│  3. Count user turns (so we can stop at 8)                          │
│  4. Call agent.run_agent_turn(messages) with a 27-second timeout    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT PIPELINE (agent.py)                                          │
│                                                                     │
│  Step A: Extract search intent                                      │
│      - Send the last few messages to llama-3.1-8b-instant (Groq)    │
│      - Get back ~10 keywords like "Java Spring SQL senior backend"  │
│                                                                     │
│  Step B: Semantic search in ChromaDB                                │
│      - Embed those keywords with all-MiniLM-L6-v2                   │
│      - Find the 20 closest assessments out of 377                   │
│                                                                     │
│  Step C: Build a prompt for the main LLM                            │
│      - System prompt = rules + mode definitions + JSON schema       │
│      - Append the 20 retrieved assessments as "CATALOG CONTEXT"     │
│      - Append the full conversation history                         │
│                                                                     │
│  Step D: Call llama-3.3-70b-versatile (Groq)                        │
│      - It picks the best 1-10 assessments, writes the reply,        │
│        and returns a JSON object                                    │
│                                                                     │
│  Step E: Parse the JSON (4-stage fallback in case it's malformed)   │
│                                                                     │
│  Step F: Validate every URL against catalog.json                    │
│      - If the model made up a URL, silently drop it                 │
│      - Guarantees zero hallucinations in the final output           │
│                                                                     │
│  Step G: Force end_of_conversation = true on turn 8 or confirmation │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RESPONSE                                                           │
│                                                                     │
│  { "reply": "...", "recommendations": [...],                        │
│    "end_of_conversation": false }                                   │
└─────────────────────────────────────────────────────────────────────┘
```

Each call goes through this whole pipeline. Roughly 1.5–2.5 seconds total.

---

## 6. Every file, what it is, why it exists

### `download_catalog.py`
- **What:** A one-time script. Fetches the SHL catalog JSON from the URL provided in the assignment and saves it as `catalog.json` locally.
- **Why:** We need a fixed snapshot of the catalog so the agent has a closed set of "valid" assessments. If the remote URL changes, we still have our copy.
- **When you run it:** Once, before deploying.

### `catalog.json`
- **What:** The ground-truth list of 377 SHL assessments. Each entry has name, URL, description, test types, etc.
- **Why:** This is the **only** source of truth. Every recommendation must come from here.

### `build_vectorstore.py`
- **What:** A one-time script. Reads `catalog.json`, converts each assessment into a sentence-transformer embedding (a 384-dimensional vector), and stores them in ChromaDB.
- **Why:** So we can do **semantic search** — "find me assessments similar to *Java developer with Spring*" — instead of literal keyword search.
- **When you run it:** Once, after `download_catalog.py`.

### `chroma_db/`
- **What:** A folder ChromaDB writes its index to. Contains a small SQLite file and the embedding vectors.
- **Why:** Persisting it means your server doesn't have to re-embed 377 items on every cold start. We ship this folder inside the Docker image.

### `agent.py`
- **What:** The brain. Has the system prompt, the search-intent extractor, the retrieval call, the main LLM call, the JSON parser, and the hallucination validator.
- **Why:** Keeping it separate from `main.py` means the agent logic can be tested without spinning up a web server.

### `main.py`
- **What:** The FastAPI app. Defines the `/health` and `/chat` endpoints. Validates incoming JSON with Pydantic. Enforces the 27-second timeout. Counts turns.
- **Why:** This is the "wrapper" that turns the agent into a callable HTTP service.

### `requirements.txt`
- **What:** Lists every Python package the project needs (FastAPI, Groq SDK, ChromaDB, sentence-transformers, etc.).
- **Why:** When Render builds your Docker image, it runs `pip install -r requirements.txt`. Without this file, the cloud machine wouldn't know what to install.

### `Dockerfile`
- **What:** A recipe that says "to run this app, start from a Python 3.11 base, install these dependencies, copy these files, expose port 8000, and run uvicorn."
- **Why:** Render builds your service inside a Docker container. The Dockerfile guarantees the cloud environment matches your local one — same Python version, same packages, same files.

### `render.yaml`
- **What:** Render's deployment config. Tells Render "use Docker, check /health for readiness, read GROQ_API_KEY from env vars, run on the free plan."
- **Why:** Lets you describe deployment in code instead of clicking around the Render dashboard.

### `.env` / `.env.example`
- **What:** Holds the GROQ_API_KEY. `.env` is gitignored (private). `.env.example` is committed as a template for others.
- **Why:** Never hardcode secrets in source code. `python-dotenv` reads `.env` at startup and puts the key into `os.environ`.

### `smoke_test.py`
- **What:** A script that hits the local server with 5 example conversations and prints the responses.
- **Why:** Quick sanity check that everything is working before submitting.

### `README.md`
- **What:** A summary of the project — stack choices, how to run, how to deploy.
- **Why:** First thing someone looking at the repo reads.

---

## 7. The three "languages" running side by side

Three different worlds touch each other in this project. It helps to know which is which.

| World | What it is | What it talks to |
|---|---|---|
| **HTTP / REST** | The way one program calls another over the network. `GET`, `POST`, JSON bodies, status codes (200 OK, 400 Bad Request, etc.). | Browser ↔ server, evaluator ↔ server, test script ↔ server |
| **Python / FastAPI** | Your application code. Reads JSON, runs logic, returns JSON. | Internally, calls the agent module, ChromaDB, Groq SDK |
| **LLM prompts** | Plain English instructions sent to the language model. | Python → Groq API → llama-3.3-70b → JSON back |

Most ML/AI courses focus on the third world (prompts, models, embeddings). The first two are the new pieces here.

---

## 8. Can I add a UI?

Yes — and it's easy because the API already exists. Three options:

### Option 1: Browser auto-generated docs (free, no work)
FastAPI auto-generates an interactive page at `http://localhost:8000/docs`. You can call `/chat` from there using a form — paste messages, click Execute, see the response. **This is what the graders use to manually test.**

### Option 2: A simple HTML chat page
Add one file `static/index.html` with an input box and a fetch() call to `/chat`. Maybe 50 lines of JavaScript. The agent doesn't change at all.

### Option 3: Streamlit / Gradio (Python-only UI)
Build a chat interface in 30 lines of Streamlit that just calls your API. Run it on Hugging Face Spaces.

For the assignment, **Option 1 is sufficient**. The SHL evaluator only cares about the API.

---

## 9. How the evaluator actually scores you

From the assignment:

> *Your endpoint is graded by an automated replay harness on our side. The harness simulates a user using an LLM that is given the trace's persona and facts and runs a real multi-turn conversation against your POST /chat.*

In plain English, the evaluator:
1. Pings `/health` (waits up to 2 minutes for your service to wake up if it's been idle).
2. Picks one of ~20 test "personas" (e.g., "I'm hiring a graduate financial analyst").
3. Starts a conversation by POSTing the first user message to `/chat`.
4. Takes your `reply`, generates the next user message using its own LLM, POSTs everything again.
5. Repeats until your `end_of_conversation: true`, or 8 turns, whichever comes first.
6. Checks the final `recommendations` against a hidden "expected shortlist" for this persona.
7. Scores you on:
   - **Schema compliance** — did every response have the right JSON shape?
   - **Recall@10** — how many "correct" assessments did you include?
   - **Behavior probes** — did you refuse off-topic? Did you ask clarifying questions on vague turn 1? Did you avoid hallucinating fake URLs?

---

## 10. How to see it running, right now

**Start the server:**
```powershell
.\.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

**Three things you can do:**

### (a) Hit the health endpoint in your browser
Open http://localhost:8000/health
You'll see: `{"status": "ok", "catalog_size": 377}`

### (b) Open the auto-generated docs UI
Open http://localhost:8000/docs

This is the **closest thing to a UI** in your project right now. You can:
- See both endpoints listed.
- Click "POST /chat" → "Try it out".
- Paste a request body like:
  ```json
  {"messages": [{"role": "user", "content": "Hiring a senior Java dev"}]}
  ```
- Click "Execute".
- See the response below.

### (c) Run the smoke test from a second terminal
```powershell
.\.venv\Scripts\python.exe smoke_test.py
```
This sends 5 test conversations and prints all the responses.

---

## 11. The one-paragraph summary

You built a small backend service. It exposes two URLs (endpoints). `/health` is a heartbeat for the host and the evaluator. `/chat` is the real work — it takes a conversation history, runs an agent that does keyword extraction → vector search → LLM generation → URL validation, and returns a structured JSON reply. The agent's job is to clarify when the recruiter is vague, recommend when there's enough context, refine on edits, compare on request, and refuse off-topic questions. Every URL it returns is verified against a local copy of the SHL catalog so it can never hallucinate. The evaluator is a program that talks to your service over HTTP, which is why the deliverable is an API and not a website. A UI can be added on top later — the auto-generated `/docs` page already gives you an interactive form for manual testing.
