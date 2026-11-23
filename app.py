"""Gradio app for the portfolio chatbot (Hugging Face Space entry point).

API endpoints used by the site widget:
  /ask(question, history) -> stream of {"answer", "sources", "done"} or {"error", "done"}
  /warmup() -> {"ready": bool, "model": str}
"""

# On ZeroGPU, `spaces` must be imported before anything initializes torch,
# so the model can be moved to "cuda" at startup (decision D21).
try:
    import spaces  # noqa: F401
except ImportError:  # running locally without the spaces package
    pass

import logging

import gradio as gr

from rag.config import load_settings
from rag.generate import HFGenerator, gpu
from rag.limits import RateLimiter, client_key
from rag.pipeline import Bot, build_index
from rag.refresh import RefreshingIndex

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("portfolio-chatbot")

settings = load_settings()

# Built now, then rebuilt from the live site every content_refresh_minutes (D27).
# If the site is unreachable, the bot answers "unavailable" and keeps retrying.
index = RefreshingIndex(lambda: build_index(settings),
                        max_age=settings.content_refresh_minutes * 60, retry_after=60)

if settings.generator == "groq":
    from rag.groq import GroqGenerator

    # Hosted API: no GPU on the Space, so visitors never hit a GPU allowance (D23).
    try:
        answer_writer = GroqGenerator(settings.gen_model, reasoning_effort=settings.reasoning_effort)
    except ValueError:
        log.exception("GROQ_API_KEY missing; answering with an 'unavailable' error")
        index, answer_writer = None, None

    @gpu(duration=10)
    def _zerogpu_placeholder():
        """Never called. ZeroGPU Spaces expect at least one GPU function to exist."""
else:
    generator = HFGenerator(settings.gen_model)

    @gpu(duration=60)
    def generate_on_gpu(messages, temperature, max_new_tokens):
        """Only this step uses the GPU, so instant replies cost no GPU time."""
        yield from generator.stream(messages, temperature, max_new_tokens)

    class GPUGenerator:
        def stream(self, messages, temperature, max_new_tokens):
            return generate_on_gpu(messages, temperature, max_new_tokens)

    answer_writer = GPUGenerator()

limiter = RateLimiter(settings.visitor_questions_per_minute, settings.visitor_questions_per_day)
bot = Bot(settings, index, answer_writer, limiter=limiter)


def ask(question, history, request: gr.Request):
    # Gradio fills in `request`; API callers only send question and history.
    host = request.client.host if request and request.client else None
    visitor = client_key(request.headers if request else {}, host)
    yield from bot.ask(question, history, client=visitor)


def warmup():
    return bot.warmup()


with gr.Blocks(title="Ask about Ariya") as demo:
    gr.Markdown(
        "# Ask about Ariya\n"
        "Answers questions about [Ariya Briscoe's portfolio](https://ryebriscoe.github.io) "
        "using only the site's content. Answers are AI-generated and may contain mistakes."
    )
    question = gr.Textbox(label="Question", placeholder="What has he built with Java?")
    history = gr.JSON(value=[], label="History ([[question, answer], ...])")
    response = gr.JSON(label="Response")
    gr.Button("Ask").click(ask, [question, history], response, api_name="ask")
    gr.Button("Warm up").click(warmup, None, response, api_name="warmup")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
