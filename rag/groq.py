"""Text generation through Groq's hosted API (decision D23).

Groq serves open models (Llama, Qwen, ...) through an OpenAI-style chat API.
Using it instead of ZeroGPU means visitors never hit a per-person GPU
allowance; the only limit is the account's daily request limit.
"""

import json
import os
from typing import Iterator

import httpx

from rag.generate import GeneratorError, drop_think_block

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

__all__ = ["GROQ_URL", "GeneratorError", "GroqGenerator"]


class GroqGenerator:
    def __init__(self, model: str, api_key: str | None = None, client: httpx.Client | None = None,
                 reasoning_effort: str | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set (add it as a secret on the Space)")
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))

    def _pieces(self, messages, temperature, max_new_tokens) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "stream": True,
        }
        if self.reasoning_effort:  # reasoning models: think briefly, show only the answer
            body["reasoning_effort"] = self.reasoning_effort
            body["include_reasoning"] = False
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with self.client.stream("POST", GROQ_URL, json=body, headers=headers) as response:
                if response.status_code != 200:
                    response.read()
                    raise GeneratorError(response.status_code, response.text[:300])
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue  # blank lines and ": keep-alive" comments
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    if "error" in chunk:  # the API can report a failure mid-stream
                        raise GeneratorError(502, str(chunk["error"])[:300])
                    choices = chunk.get("choices") or []  # e.g. a final usage-only chunk
                    delta = choices[0].get("delta", {}) if choices else {}
                    if delta.get("content"):
                        yield delta["content"]
        except httpx.HTTPError as exc:  # network failure or timeout
            raise GeneratorError(503, str(exc)[:300]) from exc

    def stream(self, messages, temperature, max_new_tokens) -> Iterator[str]:
        yield from drop_think_block(self._pieces(messages, temperature, max_new_tokens))
