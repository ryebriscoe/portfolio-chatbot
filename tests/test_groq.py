import json

import httpx
import pytest

from rag.groq import GROQ_URL, GeneratorError, GroqGenerator


def sse(*chunks, done=True):
    lines = [f"data: {json.dumps({'choices': [{'delta': c}]})}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def generator_with(handler):
    return GroqGenerator("test-model", api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))


MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]


def test_sends_an_openai_style_streaming_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse({"content": "ok"}))

    list(generator_with(handler).stream(MESSAGES, 0.2, 250))
    assert seen["url"] == GROQ_URL
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {
        "model": "test-model", "messages": MESSAGES, "temperature": 0.2,
        "max_completion_tokens": 250, "stream": True,
    }


def test_reasoning_models_get_low_effort_and_no_visible_reasoning():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse({"reasoning": "thinking..."}, {"content": "Answer."}))

    gen = GroqGenerator("openai/gpt-oss-20b", api_key="k", reasoning_effort="low",
                        client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert list(gen.stream(MESSAGES, 0.2, 1024)) == ["Answer."]
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["include_reasoning"] is False


def test_streams_the_content_pieces_in_order():
    def handler(request):
        return httpx.Response(200, content=sse({"role": "assistant"}, {"content": "He built "}, {"content": "an app."}))

    assert list(generator_with(handler).stream(MESSAGES, 0.2, 250)) == ["He built ", "an app."]


def test_stops_at_done_and_ignores_blank_and_comment_lines():
    body = b": keep-alive\n\n" + sse({"content": "A"}) + b"data: {\"choices\": [{\"delta\": {\"content\": \"late\"}}]}\n\n"
    assert list(generator_with(lambda r: httpx.Response(200, content=body)).stream(MESSAGES, 0.2, 250)) == ["A"]


def test_strips_a_leading_think_block():
    def handler(request):
        return httpx.Response(200, content=sse({"content": "<think>hmm</think>"}, {"content": "\n\nYes."}))

    assert "".join(generator_with(handler).stream(MESSAGES, 0.2, 250)) == "Yes."


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_raise_generator_error_with_the_status(status):
    gen = generator_with(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(GeneratorError) as err:
        list(gen.stream(MESSAGES, 0.2, 250))
    assert err.value.status == status


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqGenerator("test-model")


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, content=sse({"content": "x"}))

    gen = GroqGenerator("m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    list(gen.stream(MESSAGES, 0.2, 10))
    assert seen["auth"] == "Bearer from-env"


def test_in_stream_error_raises_generator_error():
    body = sse({"content": "He "}, done=False) + b'data: {"error": {"message": "overloaded"}}\n\n'
    gen = generator_with(lambda r: httpx.Response(200, content=body))
    with pytest.raises(GeneratorError):
        list(gen.stream(MESSAGES, 0.2, 250))


def test_chunks_without_choices_are_skipped():
    body = b'data: {"choices": [], "x_groq": {"usage": {}}}\n\n' + sse({"content": "ok"})
    assert list(generator_with(lambda r: httpx.Response(200, content=body)).stream(MESSAGES, 0.2, 250)) == ["ok"]


def test_network_errors_become_generator_errors():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(GeneratorError) as err:
        list(generator_with(handler).stream(MESSAGES, 0.2, 250))
    assert err.value.status == 503
