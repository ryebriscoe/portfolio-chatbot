"""Show how the bot handles one question, step by step.

    python -m rag.explain "What has he built with Java?"

Prints the chunks the search ranked highest, each score split into its
meaning and keyword parts, which chunks were kept, and the context the model
would receive. Runs on CPU and does not load the generation model.
"""

import sys

import numpy as np

from rag.config import Settings, load_settings
from rag.index import Index
from rag.prompt import format_context


def explain(index: Index, settings: Settings, question: str, extra: int = 3) -> str:
    meaning, keyword = index.score_parts(question)
    total = meaning + index.keyword_weight * keyword
    order = np.argsort(-total)[: settings.top_k + extra]

    lines = [
        f"Question: {question}",
        "",
        f"Score = meaning + {index.keyword_weight} x keyword.  "
        f"Keep the top {settings.top_k} chunks that score at least {settings.min_score}.",
        "",
        f"{'rank':<5}{'score':<7}{'meaning':<9}{'keyword':<9}{'':<9}chunk",
    ]
    kept = []
    for rank, i in enumerate(order, start=1):
        keep = rank <= settings.top_k and total[i] >= settings.min_score
        if keep:
            kept.append((index.chunks[i], float(total[i])))
        lines.append(
            f"{rank:<5}{total[i]:<7.2f}{meaning[i]:<9.2f}{keyword[i]:<9.2f}"
            f"{'kept' if keep else 'dropped':<9}{index.chunks[i].id}  ({index.chunks[i].title})"
        )
    lines.append("")
    if kept:
        lines += ["Context sent to the model:", "", format_context(kept)]
    else:
        lines.append('No chunk passed min_score, so the bot replies "not covered" without running the model.')
    return "\n".join(lines)


def main(argv=None):
    from rag.pipeline import build_index

    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = load_settings()
    print(explain(build_index(settings), settings, " ".join(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
