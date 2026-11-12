import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class Settings:
    content_url: str
    embed_model: str
    gen_model: str
    top_k: int
    min_score: float
    history_turns: int
    temperature: float
    max_new_tokens: int
    max_question_chars: int
    max_sources: int


def load_settings(path: str | os.PathLike = DEFAULT_CONFIG) -> Settings:
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f)
    for key, env in (("content_url", "CONTENT_URL"), ("gen_model", "GEN_MODEL")):
        if os.environ.get(env):
            values[key] = os.environ[env]
    return Settings(**values)
