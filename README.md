---
title: Ask About Ariya
sdk: gradio
sdk_version: "6.31.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Ask About Ariya

A chatbot that answers questions about my portfolio,
[ryebriscoe.github.io](https://ryebriscoe.github.io). It only uses what's on
the site and links to where each answer came from.

Built for the Generative AI with Large Language Models certificate
(DeepLearning.AI and AWS). Design decisions are in the
[decision log](https://github.com/ryebriscoe/ryebriscoe.github.io/blob/main/docs/rag-chatbot/decisions.md).

## How it works

1. The site publishes its content as one JSON file.
2. The app splits that content into small chunks and turns each one into numbers (embeddings).
3. For each question, it finds the 4 chunks that match best.
4. A language model writes a short answer from those chunks only.

If nothing on the site matches, it says so instead of guessing.

## Files

| File | What it does |
|---|---|
| `config.yaml` | All the settings |
| `rag/` | The chatbot code, one step per file |
| `app.py` | The web app the site talks to |
| `eval/` | 32 test questions and the script that scores them |
| `tests/` | Unit tests |

## Run it

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest tests
python -m rag.explain "What has he built with Java?"
```

`rag.explain` shows which chunks the search picked for a question and why.
