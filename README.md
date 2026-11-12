---
title: Ask About Ariya
sdk: gradio
sdk_version: "6.31.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Ask about Ariya: portfolio RAG chatbot

A retrieval-augmented chatbot that answers recruiters' questions about
[Ariya Briscoe's portfolio](https://ryebriscoe.github.io) using only the
site's own content, with links to its sources.

Built as the applied project for the **Generative AI with Large Language
Models** certificate (DeepLearning.AI and AWS). Every design decision is
recorded in the
[decision log](https://github.com/ryebriscoe/ryebriscoe.github.io/blob/main/docs/rag-chatbot/decisions.md).

## How it works

1. The Jekyll site publishes `chatbot-content.json` on every deploy.
2. At startup this app loads it, splits it into ~150 chunks, and embeds them
   with `BAAI/bge-small-en-v1.5` into an in-memory NumPy index.
3. For each question it retrieves the top 4 chunks by cosine similarity,
   builds a grounded prompt, and streams the answer from an open 7–8B
   instruct model on Hugging Face ZeroGPU.
4. Sources come from the retrieved chunks' metadata, so every link is real.

| Path | Purpose |
|---|---|
| `rag/content.py` | Load and validate the content export (retries) |
| `rag/chunker.py` | Summary + body chunks per document |
| `rag/index.py` | Embedding + NumPy cosine search |
| `rag/prompt.py` | System prompt, history, sources, input validation |
| `rag/generate.py` | Streamed generation, ZeroGPU decorator |
| `rag/pipeline.py` | validate → retrieve → prompt → stream |
| `eval/` | 30-question evaluation set and runner |
| `config.yaml` | Every tunable setting |

## Run locally

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest tests -v
set CONTENT_URL=tests/fixtures/content.json
python app.py
```
