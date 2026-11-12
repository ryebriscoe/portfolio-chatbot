"""Scoring for the evaluation set (decision D13)."""

import datetime
import statistics

from rag.prompt import is_not_covered

REFUSAL_PHRASES = ["i can only", "i can't help with", "outside what i can"]


def _norm(text: str) -> str:
    return text.lower().replace("’", "'")


def retrieval_hit(expected: list[str], retrieved_doc_ids: list[str]) -> bool:
    return not expected or any(e in retrieved_doc_ids for e in expected)


def answer_ok(answer: str, must_include: list[str], must_not_include: list[str]) -> bool:
    a = _norm(answer)
    return all(_norm(m) in a for m in must_include) and not any(_norm(m) in a for m in must_not_include)


def is_refusal(answer: str) -> bool:
    a = _norm(answer)
    return is_not_covered(answer) or any(p in a for p in REFUSAL_PHRASES)


def _fraction(flags):
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if r.get("answer") is not None]
    return {
        "retrieval_hit_rate": _fraction(r["retrieval_hit"] for r in rows if r["expected"]),
        "answer_accuracy": _fraction(r["answer_ok"] for r in answered if r["group"] in ("answerable", "status")),
        "unanswerable_clean": _fraction(
            r["refusal"] and r["answer_ok"] for r in answered if r["group"] == "unanswerable"
        ),
        "off_topic_refusal": _fraction(r["refusal"] and r["answer_ok"] for r in answered if r["group"] == "off_topic"),
        "median_ttft_s": _median(r["ttft_s"] for r in rows),
        "median_total_s": _median(r["total_s"] for r in rows),
    }


def _pct(v):
    return "—" if v is None else f"{round(v * 100)}%"


def _secs(v):
    return "—" if v is None else f"{v:.1f} s"


def to_markdown(summary: dict, label: str) -> str:
    cells = [
        datetime.date.today().isoformat(),
        label,
        summary.get("model", "—"),
        _pct(summary["retrieval_hit_rate"]),
        _pct(summary["answer_accuracy"]),
        _pct(summary["unanswerable_clean"]),
        _pct(summary["off_topic_refusal"]),
        _secs(summary["median_ttft_s"]),
        _secs(summary["median_total_s"]),
        summary.get("notes", ""),
    ]
    return "| " + " | ".join(cells) + " |"
