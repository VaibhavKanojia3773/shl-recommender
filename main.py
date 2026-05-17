"""FastAPI service for the SHL conversational assessment recommender."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shl-recommender")

ROOT = Path(__file__).parent
CATALOG_PATH = ROOT / "catalog.json"

REQUEST_TIMEOUT_SECONDS = 27.0  # leave 3s buffer under the 30s evaluator limit


# -----------------------------------------------------------------------------
# Schemas (assignment requirement — non-negotiable)
# -----------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# -----------------------------------------------------------------------------
# Lifespan: warm the singletons (collection, catalog lookup, Groq client)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("starting up — warming catalog and vector store")
    if not CATALOG_PATH.exists():
        logger.warning("catalog.json not found — service will fail to recommend")
    else:
        by_url, _, valid_urls = agent.get_catalog_lookup()
        logger.info("loaded %d assessments into catalog lookup", len(valid_urls))

    try:
        collection = agent.get_collection()
        logger.info("vector store ready (%d items)", collection.count())
    except Exception as exc:
        logger.error("failed to open ChromaDB collection: %s", exc)

    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY is not set — main model calls will fail")
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY is not set — intent extraction will fall back "
                       "to last-user-message heuristic")

    yield
    logger.info("shutting down")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    catalog_size = 0
    if CATALOG_PATH.exists():
        try:
            by_url, _, _ = agent.get_catalog_lookup()
            catalog_size = len(by_url)
        except Exception:
            catalog_size = 0
    return {"status": "ok", "catalog_size": catalog_size}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in request.messages]

    # User-turn count drives our turn limiting
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns == 0:
        raise HTTPException(status_code=400, detail="conversation must contain at least one user message")

    if user_turns > agent.MAX_TURNS:
        return ChatResponse(
            reply="We've reached the maximum conversation length. Please start a new conversation.",
            recommendations=[],
            end_of_conversation=True,
        )

    # Run the agent with a hard timeout so we never blow past the evaluator's 30s cap.
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(agent.run_agent_turn, messages),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("agent turn timed out")
        return ChatResponse(
            reply="I'm taking longer than expected. Could you rephrase or ask again?",
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception as exc:
        logger.exception("agent turn failed: %s", exc)
        return ChatResponse(
            reply="I hit an unexpected error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    return ChatResponse(**result)


# -----------------------------------------------------------------------------
# Entry point for `python main.py`
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
