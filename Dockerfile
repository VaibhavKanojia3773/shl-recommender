FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf

WORKDIR /app

# System deps for chromadb (sqlite) and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST from the PyTorch CPU index.
# This avoids pulling ~2 GB of NVIDIA/CUDA libraries that we don't use on a
# free CPU-only Render instance. Pin a version that has a manylinux CPU wheel.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1+cpu"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files. catalog.json and chroma_db/ are shipped with the image.
COPY catalog.json .
COPY chroma_db ./chroma_db
COPY agent.py main.py download_catalog.py build_vectorstore.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
