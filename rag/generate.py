"""Text generation with a Hugging Face chat model, streamed token by token."""

import os
from threading import Thread
from typing import Iterator, Protocol


class Generator(Protocol):
    def stream(self, messages: list[dict[str, str]], temperature: float, max_new_tokens: int) -> Iterator[str]:
        """Yield the answer in pieces as it is generated."""


def gpu(duration: int):
    """spaces.GPU on ZeroGPU Spaces; a no-op decorator everywhere else."""
    try:
        import spaces
    except ImportError:
        return lambda fn: fn
    return spaces.GPU(duration=duration)


class HFGenerator:
    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # On ZeroGPU the model must be moved to cuda at startup; set DEVICE=cuda
        # as a Space variable in case torch reports no GPU outside @spaces.GPU.
        self.device = device or os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)

    def stream(self, messages, temperature, max_new_tokens):
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        yield from streamer
        thread.join()
