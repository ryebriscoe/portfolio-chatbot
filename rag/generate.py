"""Text generation with a Hugging Face chat model, streamed token by token."""

import os
from threading import Thread
from typing import Iterator, Protocol


class GeneratorError(RuntimeError):
    """The model failed or refused the request. `status` is an HTTP-style code:
    429 means rate-limited (try again soon); anything else means unavailable."""

    def __init__(self, status: int, message: str):
        super().__init__(f"generation failed ({status}): {message}")
        self.status = status


class Generator(Protocol):
    def stream(self, messages: list[dict[str, str]], temperature: float, max_new_tokens: int) -> Iterator[str]:
        """Yield the answer in pieces as it is generated."""


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def chat_inputs(tokenizer, messages: list[dict[str, str]]):
    """Tokenize a chat. Thinking is turned off: Qwen3-style models otherwise
    write a long reasoning section before answering (decision D20). Templates
    without a thinking switch ignore the extra argument."""
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False
    )


def drop_think_block(pieces: Iterator[str]) -> Iterator[str]:
    """Pass streamed text through, minus a leading <think>...</think> section."""
    pieces = iter(pieces)
    buffer = ""
    for piece in pieces:
        buffer += piece
        head = buffer.lstrip()
        if len(head) < len(THINK_OPEN) and THINK_OPEN.startswith(head):
            continue  # could still become "<think>": wait for more text
        if not head.startswith(THINK_OPEN):
            yield buffer  # a normal answer: stream it unchanged
            yield from pieces
            return
        if THINK_CLOSE in head:
            break  # the thinking section is complete
    else:
        return  # the stream ended inside a thinking section

    rest = head.split(THINK_CLOSE, 1)[1].lstrip()
    while not rest:  # skip blank lines between the section and the answer
        nxt = next(pieces, None)
        if nxt is None:
            return
        rest = nxt.lstrip()
    yield rest
    yield from pieces


def gpu(duration: int):
    """spaces.GPU on ZeroGPU Spaces; a no-op decorator everywhere else."""
    try:
        import spaces
    except ImportError:
        return lambda fn: fn
    return spaces.GPU(duration=duration)


class HFGenerator:
    def __init__(self, model_name: str, device: str | None = None, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if load_in_4bit:  # evaluation on small Colab GPUs (needs bitsandbytes)
            from transformers import BitsAndBytesConfig

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto"
            )
            return
        # On ZeroGPU the model must be moved to cuda at startup; set DEVICE=cuda
        # as a Space variable in case torch reports no GPU outside @spaces.GPU.
        device = device or os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)

    def stream(self, messages, temperature, max_new_tokens):
        from transformers import TextIteratorStreamer

        inputs = chat_inputs(self.tokenizer, messages).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        yield from drop_think_block(iter(streamer))
        thread.join()
