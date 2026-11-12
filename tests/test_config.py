from rag.config import load_settings


def test_load_settings_defaults(monkeypatch):
    monkeypatch.delenv("GEN_MODEL", raising=False)
    monkeypatch.delenv("CONTENT_URL", raising=False)
    s = load_settings()
    assert s.content_url == "https://ryebriscoe.github.io/chatbot-content.json"
    assert s.embed_model == "BAAI/bge-small-en-v1.5"
    assert s.top_k == 4
    assert s.min_score == 0.35
    assert s.history_turns == 2
    assert s.temperature == 0.2
    assert s.max_new_tokens == 250
    assert s.max_question_chars == 500
    assert s.max_sources == 3


def test_env_overrides_gen_model(monkeypatch):
    monkeypatch.setenv("GEN_MODEL", "x")
    assert load_settings().gen_model == "x"


def test_env_overrides_content_url(monkeypatch):
    monkeypatch.setenv("CONTENT_URL", "local.json")
    assert load_settings().content_url == "local.json"
