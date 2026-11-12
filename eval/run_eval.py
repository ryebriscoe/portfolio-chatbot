"""Run the evaluation set and print one results-table row (decision D13).

Modes:
  retrieval  embed + search only, on CPU (no model)        -> retrieval hit rate
  local      full pipeline with gen_model in this process  -> all metrics (Colab GPU)
  space      call the deployed Space through gradio_client -> all metrics

Examples:
  python -m eval.run_eval --mode retrieval --label "baseline retrieval"
  python -m eval.run_eval --mode local --4bit --label "qwen 7b"
  python -m eval.run_eval --mode space --space user/ask-about-ariya --label "on Space"
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import yaml

from eval.scoring import answer_ok, is_refusal, retrieval_hit, summarize, to_markdown
from rag.chunker import chunk_documents
from rag.config import load_settings
from rag.content import load_documents
from rag.index import Index, SentenceTransformerEmbedder
from rag.prompt import retrieval_query

HERE = Path(__file__).resolve().parent


def load_questions(path=HERE / "questions.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _doc_ids(hits):
    return list(dict.fromkeys(c.doc_id for c, _ in hits))


def _row(q, retrieved, answer=None, ttft=None, total=None):
    row = {
        "id": q["id"], "group": q["group"], "question": q["question"],
        "expected": q["expect_sources"], "retrieved": retrieved,
        "retrieval_hit": retrieval_hit(q["expect_sources"], retrieved),
        "answer": answer, "ttft_s": ttft, "total_s": total,
        "answer_ok": None, "refusal": None,
    }
    if answer is not None:
        row["answer_ok"] = answer_ok(answer, q["must_include"], q["must_not_include"])
        row["refusal"] = is_refusal(answer)
    return row


def _timed(events):
    """Consume a stream of event dicts; return (final_event, ttft_s, total_s)."""
    start, ttft, last = time.perf_counter(), None, {}
    for event in events:
        if ttft is None and event.get("answer"):
            ttft = time.perf_counter() - start
        last = event
    return last, ttft, time.perf_counter() - start


def run_retrieval(questions, settings):
    index = Index(chunk_documents(load_documents(settings.content_url)),
                  SentenceTransformerEmbedder(settings.embed_model))
    return [
        _row(q, _doc_ids(index.search(retrieval_query(q["question"], []), settings.top_k, settings.min_score)))
        for q in questions
    ]


def run_local(questions, settings, four_bit):
    from rag.generate import HFGenerator
    from rag.pipeline import Bot

    index = Index(chunk_documents(load_documents(settings.content_url)),
                  SentenceTransformerEmbedder(settings.embed_model))
    bot = Bot(settings, index, HFGenerator(settings.gen_model, load_in_4bit=four_bit))
    rows = []
    for q in questions:
        retrieved = _doc_ids(index.search(retrieval_query(q["question"], []), settings.top_k, settings.min_score))
        final, ttft, total = _timed(bot.ask(q["question"], []))
        rows.append(_row(q, retrieved, final.get("answer", final.get("error", "")), ttft, total))
    return rows


def run_space(questions, space_id):
    from gradio_client import Client

    client = Client(space_id, hf_token=os.environ.get("HF_TOKEN"), verbose=False)
    rows = []
    for q in questions:
        final, ttft, total = _timed(client.submit(q["question"], [], api_name="/ask"))
        retrieved = [s["id"] for s in final.get("sources", [])]
        rows.append(_row(q, retrieved, final.get("answer", final.get("error", "")), ttft, total))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["retrieval", "local", "space"], required=True)
    p.add_argument("--space", help="Space id for --mode space, e.g. user/ask-about-ariya")
    p.add_argument("--label", default="")
    p.add_argument("--4bit", dest="four_bit", action="store_true", help="load gen_model in 4-bit (Colab T4)")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    settings = load_settings()
    questions = load_questions()
    if args.mode == "retrieval":
        rows, model = run_retrieval(questions, settings), "retrieval only"
    elif args.mode == "local":
        rows, model = run_local(questions, settings, args.four_bit), settings.gen_model
    else:
        if not args.space:
            p.error("--space is required with --mode space")
        rows, model = run_space(questions, args.space), f"{args.space} (Space)"

    summary = summarize(rows)
    summary["model"] = model
    summary["notes"] = f"top_k={settings.top_k}, min_score={settings.min_score}"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stamp}.json").write_text(
        json.dumps({"label": args.label, "mode": args.mode, "summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    misses = [r["id"] for r in rows if r["expected"] and not r["retrieval_hit"]]
    if misses:
        print("retrieval misses:", ", ".join(misses))
    print(to_markdown(summary, args.label))


if __name__ == "__main__":
    main()
