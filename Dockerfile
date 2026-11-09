# Container image for the AVeriTeC open-web fact-verification service.
#
# Build:  docker build -t averitec-verifier .
# Run:    docker run --rm -p 8000:8000 --env-file .env averitec-verifier
# Easier: docker compose up --build
#
# Two things this image does deliberately:
#   1. It installs requirements-service.txt, the slim runtime subset, rather
#      than requirements.txt — no sentence-transformers / torch, which only
#      the dense_rerank ablation needs. That is the difference between a
#      multi-GB image and a few hundred MB.
#   2. It bakes in the NLTK corpora, because src/bm25_baseline.py builds a
#      stopword set AT IMPORT TIME: without them the service raises
#      LookupError on startup, and the non-root user below could not download
#      them at run time anyway.

FROM python:3.12-slim

WORKDIR /app

# Dependencies first so this layer stays cached when only source changes.
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

# World-readable corpora, available offline to any user in the container.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='$NLTK_DATA', quiet=True); nltk.download('punkt', download_dir='$NLTK_DATA', quiet=True)"

COPY src/ ./src/

# Per-query DuckDuckGo cache. Mount a volume here to keep it across restarts:
# a warm cache turns ~7 s of retrieval into well under a millisecond.
ENV DDG_CACHE_DIR=/app/cache/duckduckgo

# Drop root.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p "$DDG_CACHE_DIR" \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Azure credentials are supplied at run time (compose env_file, or -e flags);
# they are never copied into the image — see .dockerignore.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "service:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
