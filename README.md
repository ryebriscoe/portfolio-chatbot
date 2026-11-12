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
2. At startup this app loads it and adds two overview documents (skills, and
   coursework by status). It splits everything into about 60 chunks and embeds
   them with `BAAI/bge-small-en-v1.5` into an in-memory NumPy index.
3. For each question it scores every chunk and keeps the top 4:

   ```
   score = meaning + 0.2 x keyword
   ```

   `meaning` is the cosine similarity of the embeddings. `keyword` is how
   many of the question's important words the chunk contains.
4. If no chunk scores at least 0.5, it replies "not covered" without running
   the model. Otherwise it builds a grounded prompt and streams the answer
   from an open instruct model on Hugging Face ZeroGPU.
5. Sources come from the retrieved chunks' metadata, so every link is real.

## Read the code in this order

Each file does one job. Reading them in this order follows a question from
start to finish.

| # | File | What it does |
|---|---|---|
| 1 | `config.yaml` | Every setting you can tune, with comments |
| 2 | `rag/content.py` | Loads the site's content file (retries 3 times) |
| 3 | `rag/overview.py` | Builds the skills and coursework overview documents |
| 4 | `rag/chunker.py` | Splits documents into chunks |
| 5 | `rag/keywords.py` | Keyword score: important words a chunk shares with the question |
| 6 | `rag/index.py` | Embeds chunks and searches them with NumPy |
| 7 | `rag/prompt.py` | The system prompt, input checks, and source links |
| 8 | `rag/generate.py` | Runs the language model and streams its answer |
| 9 | `rag/pipeline.py` | Ties it together: validate → retrieve → prompt → stream |
| 10 | `app.py` | The Gradio app the website talks to |

`eval/` holds the 32-question evaluation set and its runner.

## See it think

`rag.explain` shows what the search does with one question: the top chunks,
each score split into meaning and keyword, which chunks were kept, and the
context the model would get. It runs on CPU and doesn't load the language model.

```bash
python -m rag.explain "What has he built with Java?"
```

## Run locally

```bash
py -3.12 -m venv .venv
.venv\Scriptsctivate
pip install -r requirements-dev.txt
pytest tests -v
set CONTENT_URL=tests/fixtures/content.json
python app.py
```

## Evaluate

```bash
python -m eval.run_eval --mode retrieval --label "what I changed"
```

`retrieval` mode checks the search only and runs in seconds on CPU. `local`
and `space` modes also score the answers; see `eval/run_eval.py`.
