# APAN5560 Assignment 1

Linyi Tian | lt3039

This project adds a spaCy word embedding endpoint to the
FastAPI text generation app from Module 3.

## Run locally

Install uv and Python 3.12, then run:

```bash
uv sync
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs

## Run with Docker

With Docker running:

```bash
docker build -t sps-genai .
docker run --rm -p 127.0.0.1:8001:80 sps-genai
```

Open http://127.0.0.1:8001/docs

## Endpoints

- GET `/`: returns a greeting.
- POST `/generate`: generates text using bigram probabilities.
- GET `/embedding?word=apple`: returns the word and its full
  300-dimensional embedding using spaCy's `en_core_web_lg` model.

Example request for `/generate`:

```json
{
  "start_word": "the",
  "length": 20
}
```

Text generation may stop early if no next word is available.
Words without a vector in the spaCy model may return a zero vector.

## Checks completed

- Local `/embedding` returned HTTP 200 and a 300-dimensional vector.
- Local `/generate` returned HTTP 200 and generated text.
- The Docker image built and started successfully.
- Docker `/embedding` returned HTTP 200 and a 300-dimensional vector.