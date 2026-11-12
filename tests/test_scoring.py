from eval.scoring import answer_ok, is_refusal, retrieval_hit, summarize, to_markdown


def test_retrieval_hit():
    assert retrieval_hit(["cst-438", "cst-338"], ["cst-338", "x"]) is True
    assert retrieval_hit(["cst-438"], ["x"]) is False
    assert retrieval_hit([], ["x"]) is True


def test_answer_ok_is_case_insensitive():
    assert answer_ok("Built with spring boot and JavaFX.", ["Spring Boot", "javafx"], []) is True
    assert answer_ok("Built with Spring Boot.", ["Spring Boot", "JavaFX"], []) is False
    assert answer_ok("He completed it.", [], ["completed"]) is False


def test_is_refusal():
    assert is_refusal("That isn't covered on Ariya's portfolio. You can reach him at ...") is True
    assert is_refusal("That isn’t covered on Ariya’s portfolio.") is True
    assert is_refusal("Sorry, I can only answer questions about Ariya's work.") is True
    assert is_refusal("He built a Spring Boot app.") is False


def rows():
    return [
        {"group": "answerable", "expected": ["a"], "retrieval_hit": True, "answer": "x",
         "answer_ok": True, "refusal": False, "ttft_s": 1.0, "total_s": 3.0},
        {"group": "answerable", "expected": ["b"], "retrieval_hit": False, "answer": "x",
         "answer_ok": False, "refusal": False, "ttft_s": 2.0, "total_s": 5.0},
        {"group": "unanswerable", "expected": [], "retrieval_hit": True, "answer": "x",
         "answer_ok": False, "refusal": True, "ttft_s": 3.0, "total_s": 4.0},
        {"group": "off_topic", "expected": [], "retrieval_hit": True, "answer": "x",
         "answer_ok": True, "refusal": True, "ttft_s": 4.0, "total_s": 6.0},
    ]


def test_summarize_known_values():
    s = summarize(rows())
    assert s["retrieval_hit_rate"] == 0.5          # only rows with expectations count
    assert s["answer_accuracy"] == 0.5
    assert s["unanswerable_clean"] == 0.0          # refused, but leaked a forbidden claim
    assert s["off_topic_refusal"] == 1.0
    assert s["median_ttft_s"] == 2.5
    assert s["median_total_s"] == 4.5


def test_summarize_retrieval_only_rows_have_no_answer_metrics():
    r = [{"group": "answerable", "expected": ["a"], "retrieval_hit": True, "answer": None,
          "answer_ok": None, "refusal": None, "ttft_s": None, "total_s": None}]
    s = summarize(r)
    assert s["retrieval_hit_rate"] == 1.0
    assert s["answer_accuracy"] is None and s["median_ttft_s"] is None


def test_to_markdown_row():
    s = summarize(rows())
    s["model"] = "m"
    line = to_markdown(s, "baseline")
    assert line.startswith("| ") and line.endswith(" |")
    assert "| baseline | m | 50% | 50% | 0% | 100% | 2.5 s | 4.5 s |" in line
