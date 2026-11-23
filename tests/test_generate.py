from rag.generate import chat_inputs, drop_think_block


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [1, 2, 3]}


def test_chat_inputs_turns_thinking_off():
    tok = FakeTokenizer()
    chat_inputs(tok, [{"role": "user", "content": "hi"}])
    assert tok.kwargs["enable_thinking"] is False
    assert tok.kwargs["add_generation_prompt"] is True
    assert tok.kwargs["return_dict"] is True


def test_drop_think_block_passes_normal_answers_through():
    assert "".join(drop_think_block(iter(["He built ", "a Java app."]))) == "He built a Java app."


def test_drop_think_block_removes_a_leading_think_section():
    pieces = ["<th", "ink>\nlet me see", "...</think>", "\n\nHe built ", "a Java app."]
    assert "".join(drop_think_block(iter(pieces))) == "He built a Java app."


def test_drop_think_block_removes_an_empty_think_section():
    assert "".join(drop_think_block(iter(["<think>\n\n</think>\n\nYes."]))) == "Yes."


def test_drop_think_block_keeps_streaming_after_the_block():
    out = list(drop_think_block(iter(["<think>x</think>", "A", "B", "C"])))
    assert out[-3:] == ["A", "B", "C"]


def test_drop_think_block_handles_an_unclosed_block():
    assert "".join(drop_think_block(iter(["<think>never closes"]))) == ""
