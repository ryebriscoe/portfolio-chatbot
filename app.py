"""Gradio app for the portfolio chatbot (Hugging Face Space entry point).

API endpoints used by the site widget:
  /ask(question, history) -> stream of {"answer", "sources", "done"} or {"error", "done"}
  /warmup() -> {"ready": bool}
"""

import logging

import gradio as gr

from rag.config import load_settings
from rag.content import ContentUnavailable
from rag.generate import HFGenerator, gpu
from rag.pipeline import Bot, build_index

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("portfolio-chatbot")

settings = load_settings()

try:
    index = build_index(settings)
    log.info("indexed %d chunks", len(index.chunks))
except ContentUnavailable:
    log.exception("content unavailable; answering with an 'unavailable' error")
    index = None

generator = HFGenerator(settings.gen_model)
bot = Bot(settings, index, generator)


@gpu(duration=60)
def ask(question, history):
    yield from bot.ask(question, history)


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
